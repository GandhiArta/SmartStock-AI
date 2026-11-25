
const API_BASE_URL = "http://127.0.0.1:8001/api";

// ==========================================
// 1. UPLOAD CSV
// ==========================================
async function uploadCSVFile(fileObject) {
    const formData = new FormData();
    formData.append("file", fileObject);

    try {
        const response = await fetch(`${API_BASE_URL}/upload-csv`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                data.detail || "Gagal mengunggah file CSV."
            );
        }

        return data;
    } catch (error) {
        console.error("Upload CSV error:", error);
        alert(`Upload gagal: ${error.message}`);
        return null;
    }
}

// ==========================================
// 2. RECALCULATE FORECAST V2
// ==========================================
async function recalculateWithExogenous(items, exogenous) {
    try {
        const response = await fetch(`${API_BASE_URL}/recalculate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                items: items,
                exogenous: exogenous,
            }),
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                data.detail || "Gagal melakukan recalculation."
            );
        }

        return data;
    } catch (error) {
        console.error("Recalculate error:", error);
        alert(`Recalculate gagal: ${error.message}`);
        return null;
    }
}

// ==========================================
// 3. GET MODEL INFORMATION
// ==========================================
async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                data.detail || "Gagal mengambil informasi model."
            );
        }

        return data;
    } catch (error) {
        console.error("Model info error:", error);
        return null;
    }
}
