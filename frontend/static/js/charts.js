/**
 * Advanced Charts - Chart.js utilities and configurations
 */

// Chart.js default configuration
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#6b7280';
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.point.hoverRadius = 6;
Chart.defaults.elements.line.tension = 0.4;
Chart.defaults.elements.line.borderWidth = 2;
Chart.defaults.elements.bar.borderRadius = 4;

// Enhanced Chart Utilities
const AdvancedCharts = {
    // Color schemes
    colorSchemes: {
        primary: ['#3b82f6', '#1d4ed8', '#1e40af', '#1e3a8a'],
        success: ['#10b981', '#059669', '#047857', '#065f46'],
        warning: ['#f59e0b', '#d97706', '#b45309', '#92400e'],
        danger: ['#ef4444', '#dc2626', '#b91c1c', '#991b1b'],
        purple: ['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6'],
        rainbow: ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899']
    },

    // Get theme-aware default options
    getDefaultOptions(isDark = false) {
        const textColor = isDark ? '#d1d5db' : '#374151';
        const gridColor = isDark ? 'rgba(75, 85, 99, 0.2)' : 'rgba(209, 213, 219, 0.3)';
        const tooltipBg = isDark ? 'rgba(17, 24, 39, 0.95)' : 'rgba(255, 255, 255, 0.95)';
        const tooltipBorder = isDark ? '#374151' : '#e5e7eb';

        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += Utils.formatCurrency(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: textColor,
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: textColor,
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            return Utils.formatCurrency(value);
                        }
                    }
                }
            }
        };
    },

    // Create gradient background
    createGradient(ctx, color1, color2, direction = 'vertical') {
        const gradient = direction === 'vertical' 
            ? ctx.createLinearGradient(0, 0, 0, ctx.canvas.height)
            : ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
        
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    },

    // Create time series chart
    createTimeSeriesChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((dataset, index) => ({
                    ...dataset,
                    borderColor: dataset.borderColor || this.colorSchemes.primary[index % 4],
                    backgroundColor: dataset.backgroundColor || 
                        Utils.hexToRgba(dataset.borderColor || this.colorSchemes.primary[index % 4], 0.1),
                    fill: dataset.fill !== undefined ? dataset.fill : true,
                    tension: dataset.tension || 0.4
                }))
            },
            options: {
                ...defaultOptions,
                ...options
            }
        });
    },

    // Create forecast chart with confidence intervals
    createForecastChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        const datasets = [
            {
                label: 'Historical',
                data: data.historical,
                borderColor: this.colorSchemes.success[0],
                backgroundColor: Utils.hexToRgba(this.colorSchemes.success[0], 0.1),
                borderWidth: 3,
                fill: true
            },
            {
                label: 'Forecast',
                data: data.forecast,
                borderColor: this.colorSchemes.warning[0],
                backgroundColor: Utils.hexToRgba(this.colorSchemes.warning[0], 0.1),
                borderWidth: 3,
                borderDash: [5, 5],
                fill: false
            }
        ];

        // Add confidence intervals if provided
        if (data.confidenceUpper && data.confidenceLower) {
            datasets.push(
                {
                    label: 'Upper Confidence',
                    data: data.confidenceUpper,
                    borderColor: 'transparent',
                    backgroundColor: Utils.hexToRgba(this.colorSchemes.warning[0], 0.15),
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'Lower Confidence',
                    data: data.confidenceLower,
                    borderColor: 'transparent',
                    backgroundColor: Utils.hexToRgba(this.colorSchemes.warning[0], 0.15),
                    fill: false,
                    pointRadius: 0
                }
            );
        }

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: datasets
            },
            options: {
                ...defaultOptions,
                ...options,
                plugins: {
                    ...defaultOptions.plugins,
                    legend: {
                        ...defaultOptions.plugins.legend,
                        labels: {
                            ...defaultOptions.plugins.legend.labels,
                            filter: function(item) {
                                return !item.text.includes('Confidence');
                            }
                        }
                    }
                }
            }
        });
    },

    // Create seasonal decomposition chart
    createSeasonalChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Observed',
                        data: data.observed,
                        borderColor: this.colorSchemes.primary[0],
                        backgroundColor: 'transparent',
                        borderWidth: 2
                    },
                    {
                        label: 'Trend',
                        data: data.trend,
                        borderColor: this.colorSchemes.success[0],
                        backgroundColor: 'transparent',
                        borderWidth: 2
                    },
                    {
                        label: 'Seasonal',
                        data: data.seasonal,
                        borderColor: this.colorSchemes.purple[0],
                        backgroundColor: 'transparent',
                        borderWidth: 2
                    },
                    {
                        label: 'Residual',
                        data: data.residual,
                        borderColor: this.colorSchemes.danger[0],
                        backgroundColor: 'transparent',
                        borderWidth: 1,
                        borderDash: [3, 3]
                    }
                ]
            },
            options: {
                ...defaultOptions,
                ...options
            }
        });
    },

    // Create anomaly detection scatter plot
    createAnomalyChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        return new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal Data',
                        data: data.normal,
                        backgroundColor: this.colorSchemes.primary[0],
                        borderColor: this.colorSchemes.primary[0],
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Anomalies',
                        data: data.anomalies,
                        backgroundColor: this.colorSchemes.danger[0],
                        borderColor: this.colorSchemes.danger[0],
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }
                ]
            },
            options: {
                ...defaultOptions,
                ...options,
                scales: {
                    x: {
                        ...defaultOptions.scales.x,
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        ...defaultOptions.scales.y
                    }
                }
            }
        });
    },

    // Create performance comparison chart
    createPerformanceChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.models,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: data.accuracy,
                    backgroundColor: data.models.map((_, index) => 
                        this.colorSchemes.rainbow[index % this.colorSchemes.rainbow.length]
                    ),
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                ...defaultOptions,
                ...options,
                plugins: {
                    ...defaultOptions.plugins,
                    legend: {
                        display: false
                    }
                },
                scales: {
                    ...defaultOptions.scales,
                    y: {
                        ...defaultOptions.scales.y,
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            ...defaultOptions.scales.y.ticks,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    // Create distribution chart (histogram)
    createDistributionChart(ctx, data, options = {}) {
        const defaultOptions = this.getDefaultOptions(document.documentElement.classList.contains('dark'));
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.bins,
                datasets: [{
                    label: 'Frequency',
                    data: data.frequencies,
                    backgroundColor: this.colorSchemes.primary[0],
                    borderColor: this.colorSchemes.primary[1],
                    borderWidth: 1,
                    borderRadius: 2
                }]
            },
            options: {
                ...defaultOptions,
                ...options,
                plugins: {
                    ...defaultOptions.plugins,
                    legend: {
                        display: false
                    }
                }
            }
        });
    },

    // Update chart theme
    updateTheme(chart, isDark) {
        const newOptions = this.getDefaultOptions(isDark);
        
        // Update colors
        chart.options.plugins.legend.labels.color = newOptions.plugins.legend.labels.color;
        chart.options.plugins.tooltip.backgroundColor = newOptions.plugins.tooltip.backgroundColor;
        chart.options.plugins.tooltip.titleColor = newOptions.plugins.tooltip.titleColor;
        chart.options.plugins.tooltip.bodyColor = newOptions.plugins.tooltip.bodyColor;
        chart.options.plugins.tooltip.borderColor = newOptions.plugins.tooltip.borderColor;
        
        // Update scales
        if (chart.options.scales.x) {
            chart.options.scales.x.grid.color = newOptions.scales.x.grid.color;
            chart.options.scales.x.ticks.color = newOptions.scales.x.ticks.color;
        }
        if (chart.options.scales.y) {
            chart.options.scales.y.grid.color = newOptions.scales.y.grid.color;
            chart.options.scales.y.ticks.color = newOptions.scales.y.ticks.color;
        }
        
        chart.update();
    },

    // Export chart as image
    exportChart(chart, filename = 'chart', format = 'png') {
        const url = chart.toBase64Image();
        const link = document.createElement('a');
        link.download = `${filename}.${format}`;
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    },

    // Animate chart on load
    animateChart(chart, animationType = 'fadeIn') {
        const animations = {
            fadeIn: {
                onProgress: function(animation) {
                    chart.canvas.style.opacity = animation.currentStep / animation.numSteps;
                },
                onComplete: function() {
                    chart.canvas.style.opacity = 1;
                }
            },
            slideUp: {
                onProgress: function(animation) {
                    const progress = animation.currentStep / animation.numSteps;
                    chart.canvas.style.transform = `translateY(${20 * (1 - progress)}px)`;
                    chart.canvas.style.opacity = progress;
                },
                onComplete: function() {
                    chart.canvas.style.transform = 'translateY(0)';
                    chart.canvas.style.opacity = 1;
                }
            }
        };

        if (animations[animationType]) {
            chart.options.animation = {
                ...chart.options.animation,
                ...animations[animationType]
            };
            chart.update();
        }
    }
};

// Real-time chart updates
class RealTimeChart {
    constructor(chart, updateInterval = 5000) {
        this.chart = chart;
        this.updateInterval = updateInterval;
        this.isRunning = false;
        this.intervalId = null;
    }

    start(updateFunction) {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.intervalId = setInterval(() => {
            updateFunction(this.chart);
        }, this.updateInterval);
    }

    stop() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    updateInterval(newInterval) {
        this.updateInterval = newInterval;
        if (this.isRunning) {
            this.stop();
            this.start();
        }
    }
}

// Chart data management
class ChartDataManager {
    constructor() {
        this.datasets = new Map();
        this.cache = new Map();
    }

    addDataset(name, data) {
        this.datasets.set(name, data);
    }

    getDataset(name) {
        return this.datasets.get(name);
    }

    removeDataset(name) {
        this.datasets.delete(name);
    }

    cacheChart(name, chartData) {
        this.cache.set(name, chartData);
    }

    getCachedChart(name) {
        return this.cache.get(name);
    }

    clearCache() {
        this.cache.clear();
    }
}

// Global instances
window.AdvancedCharts = AdvancedCharts;
window.RealTimeChart = RealTimeChart;
window.ChartDataManager = ChartDataManager;

// Initialize chart data manager
window.chartDataManager = new ChartDataManager();

// Theme change listener for charts
document.addEventListener('DOMContentLoaded', function() {
    // Listen for theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'class') {
                const isDark = document.documentElement.classList.contains('dark');
                
                // Update all Chart.js instances
                Object.values(Chart.instances).forEach(chart => {
                    AdvancedCharts.updateTheme(chart, isDark);
                });
            }
        });
    });

    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class']
    });
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedCharts;
}