
let stockChartInstance = null;

// Render grafik data historis aktual dan prediksi
function renderStockChartWithData(
    skuId,
    forecastDates,
    predictions,
    historicalDates = [],
    historicalSales = []
) {
    const ctx = document.getElementById('demandChart');
    if (!ctx) return;

    // Pastikan semua input berbentuk array
    forecastDates = Array.isArray(forecastDates) ? forecastDates : [];
    predictions = Array.isArray(predictions) ? predictions : [];
    historicalDates = Array.isArray(historicalDates) ? historicalDates : [];
    historicalSales = Array.isArray(historicalSales) ? historicalSales : [];

    // Samakan panjang data historis agar tidak terjadi mismatch
    const historyLength = Math.min(
        historicalDates.length,
        historicalSales.length
    );

    historicalDates = historicalDates.slice(0, historyLength);
    historicalSales = historicalSales.slice(0, historyLength);

    // Label grafik: data historis + tanggal prediksi
    const labels = [...historicalDates, ...forecastDates];

    // Data aktual hanya ditampilkan pada periode historis
    const actualData = [
        ...historicalSales,
        ...Array(predictions.length).fill(null)
    ];

    // Data forecast dimulai dari titik aktual terakhir
    const forecastData = historyLength > 0
        ? [
            ...Array(historyLength - 1).fill(null),
            historicalSales[historyLength - 1],
            ...predictions
        ]
        : [
            ...Array(historicalDates.length).fill(null),
            ...predictions
        ];

    // Hapus grafik sebelumnya
    if (stockChartInstance) {
        stockChartInstance.destroy();
    }

    stockChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Penjualan Aktual',
                    data: actualData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2.5,
                    pointRadius: 3,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Prediksi AI (XGBoost)',
                    data: forecastData,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.05)',
                    borderDash: [5, 5],
                    borderWidth: 2.5,
                    pointRadius: 3,
                    tension: 0.3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: function (items) {
                            return items.length
                                ? items[0].label
                                : '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f1f5f9'
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}
