
// ======================================================
// GROW4STOCK AI - FRONTEND APPLICATION
// BACKEND VERSION 2 - XGBOOST
// ======================================================

let allStoresDataMap = {};
let currentStoreId = null;
let defaultExogenous = {};

// ======================================================
// HELPER: FORMAT TANGGAL
// ======================================================

function generateHistoricalDates(lastDate, length) {
    if (!lastDate || !length) return [];

    const dates = [];
    const endDate = new Date(`${lastDate}T00:00:00Z`);

    if (Number.isNaN(endDate.getTime())) return [];

    for (let i = length - 1; i >= 0; i--) {
        const date = new Date(endDate);
        date.setUTCDate(date.getUTCDate() - i);
        dates.push(date.toISOString().slice(0, 10));
    }

    return dates;
}

// ======================================================
// HELPER: MEMBACA DATA HISTORI
// Mendukung full_history berupa angka atau object
// ======================================================

function getHistoricalData(itemData) {
    let historicalDates = [];
    let historicalDemand = [];

    // Prioritaskan format histori terpisah jika tersedia.
    if (
        Array.isArray(itemData.historical_dates) &&
        Array.isArray(itemData.historical_demand)
    ) {
        historicalDates = itemData.historical_dates;
        historicalDemand = itemData.historical_demand;
    } else {
        const history = Array.isArray(itemData.full_history)
            ? itemData.full_history
            : [];

        if (history.length > 0) {
            const firstValue = history[0];

            if (typeof firstValue === "object" && firstValue !== null) {
                const validRows = history.filter(row => {
                    const value =
                        row.Demand ??
                        row.demand ??
                        row.Units ??
                        row.units ??
                        row.value;

                    return value !== undefined && Number.isFinite(Number(value));
                });

                historicalDemand = validRows.map(row =>
                    Number(
                        row.Demand ??
                        row.demand ??
                        row.Units ??
                        row.units ??
                        row.value
                    )
                );

                historicalDates = validRows.map(row =>
                    row.Date ??
                    row.date ??
                    row.timestamp ??
                    null
                );
            } else {
                historicalDemand = history
                    .map(value => Number(value))
                    .filter(value => Number.isFinite(value));
            }
        }
    }

    // Ambil 14 histori terakhir.
    historicalDemand = historicalDemand.slice(-14);

    // Jika tanggal histori tidak tersedia, turunkan dari last_date.
    if (
        historicalDates.length !== historicalDemand.length ||
        historicalDates.some(date => !date)
    ) {
        historicalDates = generateHistoricalDates(
            itemData.last_date,
            historicalDemand.length
        );
    } else {
        historicalDates = historicalDates.slice(-14).map(date =>
            String(date).slice(0, 10)
        );
    }

    return {
        historicalDates,
        historicalDemand
    };
}

// ======================================================
// MODEL INFORMATION
// ======================================================

async function loadModelInfo() {
    const info = await fetchModelInfo();

    if (!info) {
        console.warn("Informasi model belum dapat dimuat.");
        return;
    }

    updateModelInformation(info);
}

function updateModelInformation(info) {
    const accuracyElement = document.getElementById("kpiModelAccuracy");
    const accuracySubElement = document.getElementById("kpiModelAccuracySub");

    if (!accuracyElement) return;

    const wmape = info.wmape_xgboost?.wmape;

    if (wmape !== undefined && wmape !== null) {
        accuracyElement.textContent = `${Number(wmape).toFixed(2)}%`;

        if (accuracySubElement) {
            accuracySubElement.textContent =
                `WMAPE V2 | Baseline: ${Number(info.wmape_best_baseline || 0).toFixed(2)}%`;
        }
    }
}

// ======================================================
// STORE NAME MAPPING
// ======================================================

function getStoreName(storeId) {
    const storeNames = {
        S001: "Denpasar",
        S002: "Badung",
        S003: "Gianyar",
        S004: "Tabanan",
        S005: "Singaraja"
    };

    return storeNames[storeId] || storeId;
}

// ======================================================
// GET SELECTED ITEM
// ======================================================

function getSelectedItem() {
    if (!currentStoreId || !allStoresDataMap[currentStoreId]) {
        return null;
    }

    const skuSelect = document.getElementById("skuSelect");

    if (!skuSelect) return null;

    const selectedSKU = skuSelect.value;

    return allStoresDataMap[currentStoreId].find(
        item => String(item.sku_id) === String(selectedSKU)
    ) || null;
}

// ======================================================
// CHART RENDERING
// ======================================================

function renderSelectedSKUChart(itemData) {
    if (!itemData) return;

    const history = getHistoricalData(itemData);

    renderStockChartWithData(
        itemData.sku_id,
        itemData.forecast_dates || [],
        itemData.predictions || [],
        history.historicalDates,
        history.historicalDemand
    );
}

// ======================================================
// LOAD STORE DASHBOARD
// ======================================================

function loadStoreDashboard(storeId, preferredSku = null) {
    if (!storeId || !allStoresDataMap[storeId]) {
        return;
    }

    currentStoreId = storeId;

    const items = allStoresDataMap[storeId];

    const skuSelect = document.getElementById("skuSelect");

    if (!skuSelect) return;

    // Isi pilihan SKU.
    skuSelect.innerHTML = "";

    items.forEach(item => {
        const option = document.createElement("option");

        option.value = item.sku_id;
        option.textContent = item.sku_id;

        skuSelect.appendChild(option);
    });

    // Pilih SKU yang diminta atau SKU pertama.
    const selectedItem = items.find(
        item => String(item.sku_id) === String(preferredSku)
    );

    if (selectedItem) {
        skuSelect.value = selectedItem.sku_id;
    } else if (items.length > 0) {
        skuSelect.value = items[0].sku_id;
    }

    // Perbarui tabel inventory.
    updateInventoryItemsFromCSV(items);

    // Sinkronkan nilai kontrol dengan default skenario.
    syncScenarioControls();

    // Tampilkan grafik SKU terpilih.
    renderSelectedSKUChart(getSelectedItem());
}

// ======================================================
// SYNCHRONIZE SCENARIO CONTROLS
// ======================================================

function syncScenarioControls() {
    const item = getSelectedItem();

    const promotionInput = document.getElementById("scenarioPromotion");
    const discountInput = document.getElementById("scenarioDiscount");
    const weatherInput = document.getElementById("scenarioWeather");
    const epidemicInput = document.getElementById("scenarioEpidemic");

    if (!item) return;

    const defaults = item.scenario_defaults || {};

    const region = item.region;

    const defaultWeather =
        defaultExogenous.weather_by_region?.[region] ||
        defaults.weather ||
        "Sunny";

    if (promotionInput) {
        promotionInput.value = String(
            defaults.promotion ?? 0
        );
    }

    if (discountInput) {
        discountInput.value = Number(
            defaults.discount ?? 10
        );
    }

    if (weatherInput) {
        const validWeather = Array.from(weatherInput.options)
            .some(option => option.value === defaultWeather);

        weatherInput.value = validWeather
            ? defaultWeather
            : "Sunny";
    }

    if (epidemicInput) {
        epidemicInput.value = String(
            defaultExogenous.epidemic ??
            defaults.epidemic ??
            0
        );
    }
}

// ======================================================
// BUILD EXOGENOUS SCENARIO
// Format sesuai backend V2
// ======================================================

function buildExogenousScenario(items) {
    const promotion = Number(
        document.getElementById("scenarioPromotion")?.value ?? 0
    );

    const discount = Number(
        document.getElementById("scenarioDiscount")?.value ?? 0
    );

    const weather = document.getElementById("scenarioWeather")?.value || "Sunny";

    const epidemic = Number(
        document.getElementById("scenarioEpidemic")?.value ?? 0
    );

    if (!Number.isFinite(discount) || discount < 0 || discount > 100) {
        throw new Error("Discount harus berada antara 0 sampai 100.");
    }

    if (![0, 1].includes(promotion)) {
        throw new Error("Nilai Promotion tidak valid.");
    }

    if (![0, 1].includes(epidemic)) {
        throw new Error("Nilai Epidemic tidak valid.");
    }

    const weatherByRegion = {};
    const itemScenarios = {};

    items.forEach(item => {
        const region = item.region;

        if (region) {
            weatherByRegion[region] = weather;
        }

        const itemKey = `${item.store_id}|${item.sku_id}`;

        itemScenarios[itemKey] = {
            promotion: promotion,
            discount: discount
        };
    });

    return {
        epidemic: epidemic,
        weather_by_region: weatherByRegion,
        items: itemScenarios
    };
}

// ======================================================
// DOCUMENT READY
// ======================================================

document.addEventListener("DOMContentLoaded", function () {

    // --------------------------------------------------
    // ELEMENT REFERENCES
    // --------------------------------------------------

    const storeSelect = document.getElementById("storeSelect");
    const skuSelect = document.getElementById("skuSelect");

    const csvFileInput = document.getElementById("csvFileInput");
    const btnRecalculate = document.getElementById("btnRecalculate");

    const tableSearch = document.getElementById("tableSearch");
    const statusFilter = document.getElementById("statusFilter");

    const btnExportCSV = document.getElementById("btnExportCSV");

    // --------------------------------------------------
    // INITIALIZATION
    // --------------------------------------------------

    renderInventoryTable();
    loadModelInfo();

    // --------------------------------------------------
    // UPLOAD CSV
    // --------------------------------------------------

    if (csvFileInput) {
        csvFileInput.addEventListener("change", async function (event) {

            const file = event.target.files[0];

            if (!file) return;

            const response = await uploadCSVFile(file);

            if (!response) {
                csvFileInput.value = "";
                return;
            }

            if (
                !response.store_data ||
                Object.keys(response.store_data).length === 0
            ) {
                alert("Data CSV tidak memiliki informasi cabang toko.");
                csvFileInput.value = "";
                return;
            }

            // Simpan data seluruh cabang.
            allStoresDataMap = response.store_data;

            // Simpan default skenario dari backend.
            defaultExogenous = response.default_exogenous || {};

            // Perbarui informasi model.
            if (response.model_info) {
                updateModelInformation(response.model_info);
            }

            // Isi pilihan cabang.
            storeSelect.innerHTML = "";

            const stores = response.stores || Object.keys(allStoresDataMap);

            stores.forEach(storeId => {
                const option = document.createElement("option");

                option.value = storeId;
                option.textContent = `${storeId} - ${getStoreName(storeId)}`;

                storeSelect.appendChild(option);
            });

            // Pilih cabang pertama.
            const firstStore = stores[0];

            if (firstStore) {
                storeSelect.value = firstStore;
                loadStoreDashboard(firstStore);
            }

            alert(
                response.message ||
                "CSV berhasil diunggah dan diproses."
            );

            csvFileInput.value = "";
        });
    }

    // --------------------------------------------------
    // STORE SELECTOR
    // --------------------------------------------------

    if (storeSelect) {
        storeSelect.addEventListener("change", function () {

            const selectedStore = this.value;

            if (!selectedStore) return;

            loadStoreDashboard(selectedStore);
        });
    }

    // --------------------------------------------------
    // SKU SELECTOR
    // --------------------------------------------------

    if (skuSelect) {
        skuSelect.addEventListener("change", function () {

            const selectedItem = getSelectedItem();

            if (!selectedItem) return;

            syncScenarioControls();

            renderSelectedSKUChart(selectedItem);
        });
    }

    // --------------------------------------------------
    // RECALCULATE FORECAST V2
    // --------------------------------------------------

    if (btnRecalculate) {
        btnRecalculate.addEventListener("click", async function () {

            if (!currentStoreId) {
                alert("Silakan unggah CSV dan pilih cabang toko terlebih dahulu.");
                return;
            }

            const currentItems = allStoresDataMap[currentStoreId];

            if (!currentItems || currentItems.length === 0) {
                alert("Tidak ada data SKU untuk dihitung.");
                return;
            }

            const selectedSKU = document.getElementById("skuSelect")?.value;

            let exogenous;

            try {
                exogenous = buildExogenousScenario(currentItems);
            } catch (error) {
                alert(error.message);
                return;
            }

            // Konfirmasi skenario.
            const confirmed = confirm(
                "Hitung ulang forecast dengan skenario berikut?\n\n" +
                `Cabang: ${currentStoreId}\n` +
                `Promotion: ${exogenous.items[`${currentStoreId}|${selectedSKU}`]?.promotion ?? 0}\n` +
                `Discount: ${document.getElementById("scenarioDiscount")?.value}%\n` +
                `Weather: ${document.getElementById("scenarioWeather")?.value}\n` +
                `Epidemic: ${exogenous.epidemic === 1 ? "Aktif" : "Tidak aktif"}`
            );

            if (!confirmed) return;

            // Simpan teks tombol.
            const originalButtonHTML = btnRecalculate.innerHTML;

            btnRecalculate.disabled = true;
            btnRecalculate.innerHTML = `
                <i data-lucide="loader-circle" class="w-4 h-4 animate-spin"></i>
                Menghitung Forecast...
            `;

            if (window.lucide) {
                lucide.createIcons();
            }

            try {
                // Kirim data ke Backend V2.
                const result = await recalculateWithExogenous(
                    currentItems,
                    exogenous
                );

                if (!result || !Array.isArray(result.processed_items)) {
                    throw new Error(
                        "Backend tidak mengembalikan processed_items yang valid."
                    );
                }

                // Perbarui informasi model jika tersedia.
                if (result.model_info) {
                    updateModelInformation(result.model_info);
                }

                // Simpan hasil prediksi baru.
                allStoresDataMap[currentStoreId] = result.processed_items;

                // Perbarui dashboard.
                loadStoreDashboard(currentStoreId, selectedSKU);

                alert(
                    result.message ||
                    "Forecast V2 berhasil dihitung ulang."
                );

                console.log("Forecast V2 result:", result);

            } catch (error) {
                console.error("Recalculate V2 error:", error);

                alert(
                    `Gagal menghitung ulang forecast: ${error.message}`
                );

            } finally {
                btnRecalculate.disabled = false;
                btnRecalculate.innerHTML = originalButtonHTML;

                if (window.lucide) {
                    lucide.createIcons();
                }
            }
        });
    }

    // --------------------------------------------------
    // TABLE SEARCH
    // --------------------------------------------------

    if (tableSearch) {
        tableSearch.addEventListener("input", function () {
            renderInventoryTable(
                this.value,
                statusFilter?.value || "Semua Status"
            );
        });
    }

    // --------------------------------------------------
    // STATUS FILTER
    // --------------------------------------------------

    if (statusFilter) {
        statusFilter.addEventListener("change", function () {
            renderInventoryTable(
                tableSearch?.value || "",
                this.value
            );
        });
    }

    // --------------------------------------------------
    // GLOBAL SEARCH
    // --------------------------------------------------

    const globalSearch = document.querySelector(
        'header input[placeholder="Cari SKU atau nama barang..."]'
    );

    if (globalSearch) {
        globalSearch.addEventListener("input", function () {

            if (tableSearch) {
                tableSearch.value = this.value;
            }

            renderInventoryTable(
                this.value,
                statusFilter?.value || "Semua Status"
            );
        });
    }

    // --------------------------------------------------
    // EXPORT CSV
    // --------------------------------------------------

    if (btnExportCSV) {
        btnExportCSV.addEventListener("click", function () {
            exportTableToCSV();
        });
    }

    console.log("Grow4Stock AI V2 siap.");
});
