let inventoryItems = [];
let rawUploadedData = [];

function updateKPICards() {
    const kpiTotalSKU = document.getElementById('kpiTotalSKU');
    const kpiTotalSKUSub = document.getElementById('kpiTotalSKUSub');
    const kpiTotalDemand = document.getElementById('kpiTotalDemand');
    const kpiTotalDemandSub = document.getElementById('kpiTotalDemandSub');
    const kpiRiskStockout = document.getElementById('kpiRiskStockout');
    const kpiRiskStockoutSub = document.getElementById('kpiRiskStockoutSub');

    if (!kpiTotalSKU) return;

    if (inventoryItems.length === 0) {
        kpiTotalSKU.textContent = '0';
        kpiTotalSKUSub.textContent = 'Belum ada data';
        kpiTotalDemand.innerHTML = '0 <span class="text-sm font-normal text-gray-500">unit</span>';
        kpiTotalDemandSub.textContent = 'Belum ada data';
        kpiRiskStockout.textContent = '0 SKU';
        kpiRiskStockout.className = 'text-2xl font-black text-gray-400 mt-1';
        kpiRiskStockoutSub.textContent = 'Stok aman';
        return;
    }

    const totalSKU = inventoryItems.length;
    const totalDemand = inventoryItems.reduce((acc, item) => acc + (item.demand7d || 0), 0);
    const riskCount = inventoryItems.filter(item => item.status === 'Risk Stockout').length;

    kpiTotalSKU.textContent = totalSKU.toLocaleString('id-ID');
    kpiTotalSKUSub.textContent = `+${totalSKU} SKU diunggah`;

    kpiTotalDemand.innerHTML = `${totalDemand.toLocaleString('id-ID')} <span class="text-sm font-normal text-gray-500">unit</span>`;
    kpiTotalDemandSub.textContent = `Total kalkulasi 7 hari`;

    kpiRiskStockout.textContent = `${riskCount} SKU`;
    if (riskCount > 0) {
        kpiRiskStockout.className = 'text-2xl font-black text-rose-600 mt-1';
        kpiRiskStockoutSub.textContent = 'Perlu Reorder segera';
    } else {
        kpiRiskStockout.className = 'text-2xl font-black text-emerald-600 mt-1';
        kpiRiskStockoutSub.textContent = 'Seluruh stok aman';
    }
}

function renderInventoryTable(filterQuery = '', filterStatus = 'Semua Status') {
    updateKPICards();

    const tableBody = document.querySelector('table tbody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    if (inventoryItems.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="9" class="py-12 text-center text-gray-400">
                    <div class="flex flex-col items-center justify-center gap-2">
                        <i data-lucide="file-spreadsheet" class="w-8 h-8 text-gray-300"></i>
                        <p class="font-medium">Belum ada data transaksi yang diunggah.</p>
                        <p class="text-xs text-gray-400">Silakan unggah file CSV transaksi toko Anda untuk melihat rekomendasi ROP.</p>
                    </div>
                </td>
            </tr>
        `;
        if (window.lucide) lucide.createIcons();
        return;
    }

    const filtered = inventoryItems.filter(item => {
        const matchesQuery = item.name.toLowerCase().includes(filterQuery.toLowerCase()) || 
                             item.sku.toLowerCase().includes(filterQuery.toLowerCase());
        const matchesStatus = filterStatus === 'Semua Status' || item.status === filterStatus;
        return matchesQuery && matchesStatus;
    });

    if (filtered.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="9" class="py-6 text-center text-gray-400">Data barang tidak ditemukan.</td>
            </tr>
        `;
        return;
    }

    filtered.forEach(item => {
        const poQty = item.suggestedPO;
        
        let badgeClass = 'bg-emerald-100 text-emerald-700';
        if (item.status === 'Risk Stockout') badgeClass = 'bg-rose-100 text-rose-700';
        if (item.status === 'Overstock') badgeClass = 'bg-amber-100 text-amber-700';

        const row = `
            <tr class="border-b border-gray-100 hover:bg-gray-50 transition">
                <td class="py-3 px-4 font-medium text-gray-800">${item.sku}</td>
                <td class="py-3 px-4 text-gray-600">${item.category}</td>
                <td class="py-3 px-4 text-gray-800 font-semibold">${item.stock.toLocaleString('id-ID')} unit</td>
                <td class="py-3 px-4 text-gray-600">${item.safety_stock.toLocaleString('id-ID')} unit</td>
                <td class="py-3 px-4 text-gray-600 font-medium">${item.rop.toLocaleString('id-ID')} unit</td>
                <td class="py-3 px-4 font-semibold text-gray-800">${item.demand7d.toLocaleString('id-ID')} unit</td>
                <td class="py-3 px-4 font-bold text-indigo-600">${poQty > 0 ? poQty.toLocaleString('id-ID') + ' unit' : '-'}</td>
                <td class="py-3 px-4">
                    <span class="px-2.5 py-1 text-xs font-semibold rounded-full ${badgeClass}">
                        ${item.status}
                    </span>
                </td>
                <td class="py-3 px-4">
                    <button onclick="openPOModal('${item.sku}', ${poQty})" class="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-lg transition shadow-sm">
                        Buat PO
                    </button>
                </td>
            </tr>
        `;
        tableBody.innerHTML += row;
    });

    if (window.lucide) lucide.createIcons();
}

function updateInventoryItemsFromCSV(processedItems) {
    rawUploadedData = processedItems;

    inventoryItems = processedItems.map((item) => {
        const pol = item.inventory_policy || {};
        const preds = item.predictions || [];
        
        const demand7d = preds.length >= 7 
            ? Math.round(preds.slice(0, 7).reduce((sum, val) => sum + val, 0)) 
            : 0;

        const stock = item.current_stock !== undefined ? item.current_stock : 0;
        const rop = pol.reorder_point || 0;
        const safety_stock = pol.safety_stock || 0;
        const suggestedPO = pol.suggested_order_qty !== undefined 
            ? pol.suggested_order_qty 
            : (stock <= rop ? Math.max(0, rop - stock) : 0);

        let status = 'Optimal';
        if (stock <= rop) {
            status = 'Risk Stockout';
        } else if (stock > rop * 2.5) {
            status = 'Overstock';
        }

        return {
            sku: item.sku_id,
            name: item.sku_id,
            category: item.category || 'Retail',
            stock: stock,
            safety_stock: safety_stock,
            rop: rop,
            demand7d: demand7d,
            suggestedPO: suggestedPO,
            status: status
        };
    });

    renderInventoryTable();
}

function openPOModal(sku, qty) {
    if (qty <= 0) {
        alert(`Stok untuk ${sku} masih mencukupi (di atas Reorder Point). Belum memerlukan Purchase Order.`);
        return;
    }
    alert(`[PURCHASE ORDER DIBUAT]\n• SKU: ${sku}\n• Jumlah Pemesanan (PO): ${qty.toLocaleString('id-ID')} unit\n• Target Kedatangan: +3 Hari (Lead Time)`);
}

function exportTableToCSV() {
    if (inventoryItems.length === 0) {
        alert("Tidak ada data untuk diekspor. Silakan unggah CSV terlebih dahulu.");
        return;
    }
    let csvContent = "data:text/csv;charset=utf-8,SKU,Kategori,Stok Saat Ini,Safety Stock,ROP,Prediksi Demand 7D,Rekomendasi PO,Status\n";
    inventoryItems.forEach(item => {
        csvContent += `${item.sku},${item.category},${item.stock},${item.safety_stock},${item.rop},${item.demand7d},${item.suggestedPO},${item.status}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "Rekomendasi_Restock_ROP.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}