/**
 * Dashboard-specific JavaScript functionality
 */

// Dashboard state management
const DashboardState = {
    charts: new Map(),
    kpis: {},
    refreshInterval: null,
    isLoading: false,
    lastUpdate: null
};

// Dashboard initialization
function initializeDashboard() {
    perfMonitor.mark('dashboard-init-start');
    
    // Load dashboard data
    loadDashboardData();
    
    // Setup auto-refresh
    setupAutoRefresh();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
    
    perfMonitor.mark('dashboard-init-end');
    perfMonitor.measure('dashboard-initialization', 'dashboard-init-start', 'dashboard-init-end');
}

// Load all dashboard data
async function loadDashboardData() {
    try {
        DashboardState.isLoading = true;
        showLoading();
        
        // Load KPIs
        await loadKPIs();
        
        // Load chart data
        await Promise.all([
            loadSalesTrendData(),
            loadForecastData(),
            loadSeasonalData()
        ]);
        
        DashboardState.lastUpdate = new Date();
        updateLastRefreshTime();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showToast('Error loading dashboard data', 'error');
    } finally {
        DashboardState.isLoading = false;
        hideLoading();
    }
}

// Load KPIs from API
async function loadKPIs() {
    try {
        const response = await api.get('/analytics/kpis');
        
        if (response.success) {
            DashboardState.kpis = response.kpis;
            updateKPICards(response.kpis);
            animateKPICards();
        }
    } catch (error) {
        console.error('Error loading KPIs:', error);
        // Use fallback data
        updateKPICards(generateFallbackKPIs());
    }
}

// Update KPI cards with animation
function updateKPICards(kpis) {
    // Current Month Sales
    animateNumber('current-sales', 0, parseFloat(kpis.current_month_sales.value), 1000, (value) => 
        Utils.formatCurrency(value));
    
    // MoM Change
    updateMoMChange(kpis.mom_growth);
    
    // YoY Growth
    animateNumber('yoy-growth', 0, parseFloat(kpis.yoy_growth.value), 1000, (value) => 
        value.toFixed(1) + '%');
    
    // Forecast Accuracy
    animateNumber('forecast-accuracy', 0, parseFloat(kpis.forecast_accuracy.value), 1000, (value) => 
        value.toFixed(1) + '%');
    
    // Trend Direction
    updateTrendDirection(kpis.trend_direction);
}

// Animate KPI cards entrance
function animateKPICards() {
    const kpiCards = document.querySelectorAll('.kpi-card, [class*="bg-white"][class*="rounded-xl"]');
    
    kpiCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Animate number changes
function animateNumber(elementId, start, end, duration, formatter) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startTime = performance.now();
    const range = end - start;
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (easeOutCubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        
        const currentValue = start + (range * easeProgress);
        element.textContent = formatter ? formatter(currentValue) : currentValue.toFixed(0);
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Update MoM change indicator
function updateMoMChange(momGrowth) {
    const element = document.getElementById('mom-change');
    if (!element) return;
    
    const value = parseFloat(momGrowth.value);
    const trend = momGrowth.trend;
    
    let iconClass, textClass;
    if (trend === 'up') {
        iconClass = 'fa-arrow-up text-success-500';
        textClass = 'text-success-600 dark:text-success-400';
    } else if (trend === 'down') {
        iconClass = 'fa-arrow-down text-danger-500';
        textClass = 'text-danger-600 dark:text-danger-400';
    } else {
        iconClass = 'fa-arrow-right text-gray-500';
        textClass = 'text-gray-500 dark:text-gray-400';
    }
    
    element.className = `text-sm ${textClass} mt-1`;
    element.innerHTML = `
        <span class="inline-flex items-center">
            <i class="fas ${iconClass} mr-1"></i>
            ${Math.abs(value).toFixed(1)}% MoM
        </span>
    `;
}

// Update trend direction
function updateTrendDirection(trendDirection) {
    const element = document.getElementById('trend-direction');
    const iconElement = document.getElementById('trend-icon');
    
    if (!element || !iconElement) return;
    
    element.textContent = trendDirection.formatted;
    
    const direction = trendDirection.value;
    if (direction === 'increasing') {
        iconElement.className = 'fas fa-arrow-up text-success-600 dark:text-success-400 text-xl';
    } else if (direction === 'decreasing') {
        iconElement.className = 'fas fa-arrow-down text-danger-600 dark:text-danger-400 text-xl';
    } else {
        iconElement.className = 'fas fa-arrow-right text-gray-600 dark:text-gray-400 text-xl';
    }
}

// Load sales trend data and create chart
async function loadSalesTrendData() {
    try {
        // For demo purposes, using sample data
        // In production, this would call: const response = await api.get('/analytics/trends');
        const sampleData = generateSampleSalesData(24);
        
        createSalesTrendChart(sampleData);
        
    } catch (error) {
        console.error('Error loading sales trend data:', error);
        // Show placeholder chart
        createSalesTrendChart(generateSampleSalesData(24));
    }
}

// Create sales trend chart
function createSalesTrendChart(data) {
    const ctx = document.getElementById('sales-trend-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (DashboardState.charts.has('salesTrend')) {
        DashboardState.charts.get('salesTrend').destroy();
    }
    
    const chart = AdvancedCharts.createTimeSeriesChart(ctx, {
        labels: data.labels,
        datasets: [{
            label: 'Sales',
            data: data.sales,
            borderColor: AdvancedCharts.colorSchemes.primary[0],
            backgroundColor: Utils.hexToRgba(AdvancedCharts.colorSchemes.primary[0], 0.1),
            fill: true
        }]
    });
    
    DashboardState.charts.set('salesTrend', chart);
    
    // Add animation
    AdvancedCharts.animateChart(chart, 'fadeIn');
}

// Load forecast data and create chart
async function loadForecastData() {
    try {
        // Sample forecast data
        const historicalData = generateSampleSalesData(18);
        const forecastData = generateSampleForecastData(6);
        
        createForecastChart(historicalData, forecastData);
        
    } catch (error) {
        console.error('Error loading forecast data:', error);
    }
}

// Create forecast vs actual chart
function createForecastChart(historical, forecast) {
    const ctx = document.getElementById('forecast-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (DashboardState.charts.has('forecast')) {
        DashboardState.charts.get('forecast').destroy();
    }
    
    const allLabels = [...historical.labels, ...forecast.labels];
    
    const chart = AdvancedCharts.createForecastChart(ctx, {
        labels: allLabels,
        historical: [...historical.sales, ...Array(forecast.labels.length).fill(null)],
        forecast: [...Array(historical.labels.length).fill(null), ...forecast.forecast],
        confidenceUpper: [...Array(historical.labels.length).fill(null), ...forecast.upper],
        confidenceLower: [...Array(historical.labels.length).fill(null), ...forecast.lower]
    });
    
    DashboardState.charts.set('forecast', chart);
}

// Load seasonal data and create chart
async function loadSeasonalData() {
    try {
        const monthlyData = [
            2800, 2600, 2700, 2900, 3100, 3300,
            3200, 3400, 3600, 3800, 4200, 4500
        ];
        
        createSeasonalChart(monthlyData);
        
    } catch (error) {
        console.error('Error loading seasonal data:', error);
    }
}

// Create seasonal patterns chart
function createSeasonalChart(monthlyData) {
    const ctx = document.getElementById('seasonal-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (DashboardState.charts.has('seasonal')) {
        DashboardState.charts.get('seasonal').destroy();
    }
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: 'Average Monthly Sales',
                data: monthlyData,
                backgroundColor: AdvancedCharts.colorSchemes.rainbow,
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            ...AdvancedCharts.getDefaultOptions(document.documentElement.classList.contains('dark')),
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000).toFixed(1) + 'K';
                        }
                    }
                }
            }
        }
    });
    
    DashboardState.charts.set('seasonal', chart);
}

// Time range change handler
function changeTimeRange(range) {
    // Update active button
    document.querySelectorAll('.time-range-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update chart data based on range
    let months;
    switch(range) {
        case '6M': months = 6; break;
        case '1Y': months = 12; break;
        case '2Y': months = 24; break;
        default: months = 48; break;
    }
    
    const newData = generateSampleSalesData(months);
    const salesChart = DashboardState.charts.get('salesTrend');
    
    if (salesChart) {
        salesChart.data.labels = newData.labels;
        salesChart.data.datasets[0].data = newData.sales;
        salesChart.update('active');
    }
    
    showToast(`Updated chart to show ${range} data`, 'success');
}

// Refresh forecast data
function refreshForecast() {
    showLoading();
    
    // Simulate API call delay
    setTimeout(() => {
        const historicalData = generateSampleSalesData(18);
        const forecastData = generateSampleForecastData(6);
        
        const forecastChart = DashboardState.charts.get('forecast');
        if (forecastChart) {
            const allLabels = [...historicalData.labels, ...forecastData.labels];
            
            forecastChart.data.labels = allLabels;
            forecastChart.data.datasets[0].data = [...historicalData.sales, ...Array(forecastData.labels.length).fill(null)];
            forecastChart.data.datasets[1].data = [...Array(historicalData.labels.length).fill(null), ...forecastData.forecast];
            
            if (forecastChart.data.datasets[2]) {
                forecastChart.data.datasets[2].data = [...Array(historicalData.labels.length).fill(null), ...forecastData.upper];
            }
            if (forecastChart.data.datasets[3]) {
                forecastChart.data.datasets[3].data = [...Array(historicalData.labels.length).fill(null), ...forecastData.lower];
            }
            
            forecastChart.update('active');
        }
        
        hideLoading();
        showToast('Forecast updated successfully', 'success');
    }, 1000);
}

// Setup auto-refresh functionality
function setupAutoRefresh() {
    // Refresh every 5 minutes
    DashboardState.refreshInterval = setInterval(() => {
        if (!DashboardState.isLoading) {
            loadDashboardData();
        }
    }, 5 * 60 * 1000);
}

// Initialize real-time updates
function initializeRealTimeUpdates() {
    // Simulate real-time data updates
    setInterval(() => {
        if (!DashboardState.isLoading) {
            updateRealTimeMetrics();
        }
    }, 30000); // Update every 30 seconds
}

// Update real-time metrics
function updateRealTimeMetrics() {
    // Simulate small changes in KPIs
    if (DashboardState.kpis.current_month_sales) {
        const currentValue = parseFloat(DashboardState.kpis.current_month_sales.value);
        const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
        const newValue = currentValue * (1 + variation);
        
        DashboardState.kpis.current_month_sales.value = newValue;
        DashboardState.kpis.current_month_sales.formatted = Utils.formatCurrency(newValue);
        
        // Update display with subtle animation
        animateNumber('current-sales', currentValue, newValue, 500, Utils.formatCurrency);
    }
}

// Update last refresh time display
function updateLastRefreshTime() {
    const lastUpdateElement = document.querySelector('.last-update');
    if (lastUpdateElement && DashboardState.lastUpdate) {
        lastUpdateElement.textContent = `Last updated: ${DashboardState.lastUpdate.toLocaleTimeString()}`;
    }
}

// Generate sample data functions
function generateSampleSalesData(months = 24) {
    const labels = [];
    const sales = [];
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - months);
    
    for (let i = 0; i < months; i++) {
        const date = new Date(startDate);
        date.setMonth(date.getMonth() + i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
        
        // Generate realistic sales data with trend and seasonality
        const trend = 2000 + (i * 50);
        const seasonal = 500 * Math.sin((i % 12) * Math.PI / 6);
        const noise = (Math.random() - 0.5) * 300;
        sales.push(Math.round(trend + seasonal + noise));
    }
    
    return { labels, sales };
}

function generateSampleForecastData(months = 6) {
    const labels = [];
    const forecast = [];
    const upper = [];
    const lower = [];
    const startDate = new Date();
    
    for (let i = 1; i <= months; i++) {
        const date = new Date(startDate);
        date.setMonth(date.getMonth() + i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
        
        const baseValue = 4500 + (i * 100);
        const seasonal = 300 * Math.sin((i % 12) * Math.PI / 6);
        const forecastValue = Math.round(baseValue + seasonal);
        
        forecast.push(forecastValue);
        upper.push(Math.round(forecastValue * 1.15));
        lower.push(Math.round(forecastValue * 0.85));
    }
    
    return { labels, forecast, upper, lower };
}

function generateFallbackKPIs() {
    return {
        current_month_sales: {
            value: 4250,
            formatted: '$4,250'
        },
        mom_growth: {
            value: 8.5,
            formatted: '+8.5%',
            trend: 'up'
        },
        yoy_growth: {
            value: 12.3,
            formatted: '+12.3%',
            trend: 'up'
        },
        forecast_accuracy: {
            value: 89.2,
            formatted: '89.2%'
        },
        trend_direction: {
            value: 'increasing',
            formatted: 'Increasing'
        }
    };
}

// Dashboard cleanup
function cleanupDashboard() {
    // Clear intervals
    if (DashboardState.refreshInterval) {
        clearInterval(DashboardState.refreshInterval);
    }
    
    // Destroy charts
    DashboardState.charts.forEach(chart => {
        chart.destroy();
    });
    DashboardState.charts.clear();
}

// Quick action handlers
function startForecast() {
    window.location.href = '/forecast';
}

function uploadData() {
    window.location.href = '/data-management';
}

function viewAnalytics() {
    window.location.href = '/analytics';
}

function exportData() {
    showToast('Export functionality coming soon', 'info');
}

// Responsive chart handling
function handleResize() {
    DashboardState.charts.forEach(chart => {
        chart.resize();
    });
}

// Event listeners
window.addEventListener('resize', Utils.throttle(handleResize, 250));

// Cleanup on page unload
window.addEventListener('beforeunload', cleanupDashboard);

// Export for global use
window.DashboardState = DashboardState;
window.initializeDashboard = initializeDashboard;
window.changeTimeRange = changeTimeRange;
window.refreshForecast = refreshForecast;
window.startForecast = startForecast;
window.uploadData = uploadData;
window.viewAnalytics = viewAnalytics;
window.exportData = exportData;