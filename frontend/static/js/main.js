/**
 * Advanced Sales Forecasting Platform - Main JavaScript
 * Core functionality and utilities
 */

// Global configuration
const CONFIG = {
    API_BASE_URL: '/api',
    CHART_COLORS: {
        primary: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#06b6d4',
        purple: '#8b5cf6',
        pink: '#ec4899',
        gray: '#6b7280'
    },
    ANIMATION_DURATION: 300,
    TOAST_DURATION: 3000
};

// Global state management
const AppState = {
    currentModel: localStorage.getItem('selectedModel') || 'ensemble',
    theme: localStorage.getItem('theme') || 'light',
    user: {
        name: 'Admin',
        preferences: {}
    },
    data: {
        lastUpdated: null,
        cache: new Map()
    }
};

// API Client
class APIClient {
    constructor(baseURL = CONFIG.API_BASE_URL) {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            showLoading();
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        } finally {
            hideLoading();
        }
    }

    // GET request
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url);
    }

    // POST request
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // PUT request
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    // DELETE request
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    // File upload
    async uploadFile(endpoint, file, additionalData = {}) {
        const formData = new FormData();
        formData.append('file', file);
        
        Object.keys(additionalData).forEach(key => {
            formData.append(key, additionalData[key]);
        });

        return this.request(endpoint, {
            method: 'POST',
            body: formData,
            headers: {} // Remove Content-Type to let browser set it for FormData
        });
    }
}

// Initialize API client
const api = new APIClient();

// Utility Functions
const Utils = {
    // Format numbers
    formatNumber(num, decimals = 0) {
        if (num === null || num === undefined) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(num);
    },

    // Format currency
    formatCurrency(amount, currency = 'USD') {
        if (amount === null || amount === undefined) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },

    // Format percentage
    formatPercentage(value, decimals = 1) {
        if (value === null || value === undefined) return 'N/A';
        return `${value.toFixed(decimals)}%`;
    },

    // Format date
    formatDate(date, options = {}) {
        if (!date) return 'N/A';
        const dateObj = typeof date === 'string' ? new Date(date) : date;
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            ...options
        }).format(dateObj);
    },

    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Generate random color
    generateColor(opacity = 1) {
        const colors = Object.values(CONFIG.CHART_COLORS);
        const color = colors[Math.floor(Math.random() * colors.length)];
        return opacity === 1 ? color : this.hexToRgba(color, opacity);
    },

    // Convert hex to rgba
    hexToRgba(hex, alpha = 1) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },

    // Validate email
    validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },

    // Generate UUID
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    },

    // Copy to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            showToast('Copied to clipboard', 'success');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            showToast('Failed to copy to clipboard', 'error');
        }
    },

    // Download file
    downloadFile(data, filename, type = 'application/octet-stream') {
        const blob = new Blob([data], { type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    },

    // Storage helpers
    storage: {
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
            } catch (error) {
                console.error('Storage set error:', error);
            }
        },

        get(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.error('Storage get error:', error);
                return defaultValue;
            }
        },

        remove(key) {
            try {
                localStorage.removeItem(key);
            } catch (error) {
                console.error('Storage remove error:', error);
            }
        },

        clear() {
            try {
                localStorage.clear();
            } catch (error) {
                console.error('Storage clear error:', error);
            }
        }
    }
};

// Loading Management
let loadingCount = 0;

function showLoading() {
    loadingCount++;
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('hidden');
    }
}

function hideLoading() {
    loadingCount = Math.max(0, loadingCount - 1);
    if (loadingCount === 0) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
}

// Toast Notification System
function showToast(message, type = 'info', duration = CONFIG.TOAST_DURATION) {
    const container = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    const id = Utils.generateUUID();
    toast.id = `toast-${id}`;
    
    const bgColors = {
        success: 'bg-green-500',
        error: 'bg-red-500',
        warning: 'bg-yellow-500',
        info: 'bg-blue-500'
    };
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.className = `${bgColors[type] || bgColors.info} text-white px-6 py-4 rounded-lg shadow-lg transform translate-x-full transition-all duration-300 ease-in-out mb-2 max-w-sm`;
    
    toast.innerHTML = `
        <div class="flex items-center justify-between">
            <div class="flex items-center">
                <i class="fas ${icons[type] || icons.info} mr-3"></i>
                <span class="font-medium">${message}</span>
            </div>
            <button onclick="removeToast('${id}')" class="ml-4 text-white hover:text-gray-200 focus:outline-none">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-x-full');
    }, 100);
    
    // Auto remove
    setTimeout(() => {
        removeToast(id);
    }, duration);
    
    return id;
}

function removeToast(id) {
    const toast = document.getElementById(`toast-${id}`);
    if (toast) {
        toast.classList.add('translate-x-full');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'fixed top-4 right-4 z-50 space-y-2';
    document.body.appendChild(container);
    return container;
}

// Modal Management
class Modal {
    constructor(options = {}) {
        this.options = {
            title: 'Modal',
            content: '',
            size: 'md', // sm, md, lg, xl
            closable: true,
            backdrop: true,
            ...options
        };
        this.element = null;
        this.isOpen = false;
    }

    create() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content max-w-${this.options.size}">
                <div class="modal-header">
                    <h3 class="modal-title">${this.options.title}</h3>
                    ${this.options.closable ? '<button class="modal-close text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"><i class="fas fa-times"></i></button>' : ''}
                </div>
                <div class="modal-body">
                    ${this.options.content}
                </div>
                ${this.options.footer ? `<div class="modal-footer">${this.options.footer}</div>` : ''}
            </div>
        `;

        // Event listeners
        if (this.options.closable) {
            const closeBtn = modal.querySelector('.modal-close');
            closeBtn.addEventListener('click', () => this.close());
        }

        if (this.options.backdrop) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.close();
                }
            });
        }

        this.element = modal;
        return modal;
    }

    open() {
        if (this.isOpen) return;
        
        if (!this.element) {
            this.create();
        }

        document.body.appendChild(this.element);
        document.body.style.overflow = 'hidden';
        this.isOpen = true;

        // Trigger custom event
        window.dispatchEvent(new CustomEvent('modalOpen', { detail: this }));
    }

    close() {
        if (!this.isOpen) return;

        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        
        document.body.style.overflow = '';
        this.isOpen = false;

        // Trigger custom event
        window.dispatchEvent(new CustomEvent('modalClose', { detail: this }));
    }

    updateContent(content) {
        if (this.element) {
            const body = this.element.querySelector('.modal-body');
            if (body) {
                body.innerHTML = content;
            }
        }
    }
}

// Form Validation
class FormValidator {
    constructor(form, rules = {}) {
        this.form = typeof form === 'string' ? document.querySelector(form) : form;
        this.rules = rules;
        this.errors = {};
    }

    validate() {
        this.errors = {};
        
        Object.keys(this.rules).forEach(fieldName => {
            const field = this.form.querySelector(`[name="${fieldName}"]`);
            if (!field) return;

            const value = field.value.trim();
            const fieldRules = this.rules[fieldName];

            fieldRules.forEach(rule => {
                if (this.errors[fieldName]) return; // Skip if already has error

                switch (rule.type) {
                    case 'required':
                        if (!value) {
                            this.addError(fieldName, rule.message || 'This field is required');
                        }
                        break;
                    
                    case 'email':
                        if (value && !Utils.validateEmail(value)) {
                            this.addError(fieldName, rule.message || 'Please enter a valid email');
                        }
                        break;
                    
                    case 'min':
                        if (value && value.length < rule.value) {
                            this.addError(fieldName, rule.message || `Minimum ${rule.value} characters required`);
                        }
                        break;
                    
                    case 'max':
                        if (value && value.length > rule.value) {
                            this.addError(fieldName, rule.message || `Maximum ${rule.value} characters allowed`);
                        }
                        break;
                    
                    case 'custom':
                        if (!rule.validator(value)) {
                            this.addError(fieldName, rule.message || 'Invalid value');
                        }
                        break;
                }
            });
        });

        this.displayErrors();
        return Object.keys(this.errors).length === 0;
    }

    addError(field, message) {
        this.errors[field] = message;
    }

    displayErrors() {
        // Clear previous errors
        this.form.querySelectorAll('.error-message').forEach(el => el.remove());
        this.form.querySelectorAll('.border-red-500').forEach(el => {
            el.classList.remove('border-red-500');
            el.classList.add('border-gray-300', 'dark:border-gray-600');
        });

        // Display new errors
        Object.keys(this.errors).forEach(fieldName => {
            const field = this.form.querySelector(`[name="${fieldName}"]`);
            if (field) {
                field.classList.remove('border-gray-300', 'dark:border-gray-600');
                field.classList.add('border-red-500');

                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message text-red-600 text-sm mt-1';
                errorDiv.textContent = this.errors[fieldName];
                field.parentNode.appendChild(errorDiv);
            }
        });
    }
}

// Chart Utilities
const ChartUtils = {
    // Default chart options
    getDefaultOptions(isDark = false) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: isDark ? '#d1d5db' : '#374151',
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: isDark ? 'rgba(17, 24, 39, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                    titleColor: isDark ? '#f3f4f6' : '#111827',
                    bodyColor: isDark ? '#f3f4f6' : '#111827',
                    borderColor: isDark ? '#374151' : '#e5e7eb',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: isDark ? 'rgba(75, 85, 99, 0.2)' : 'rgba(209, 213, 219, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: isDark ? '#9ca3af' : '#6b7280'
                    }
                },
                y: {
                    grid: {
                        color: isDark ? 'rgba(75, 85, 99, 0.2)' : 'rgba(209, 213, 219, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: isDark ? '#9ca3af' : '#6b7280'
                    }
                }
            }
        };
    },

    // Create gradient
    createGradient(ctx, color1, color2, direction = 'vertical') {
        const gradient = direction === 'vertical' 
            ? ctx.createLinearGradient(0, 0, 0, ctx.canvas.height)
            : ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
        
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    },

    // Format chart data
    formatChartData(labels, datasets) {
        return {
            labels,
            datasets: datasets.map((dataset, index) => ({
                ...dataset,
                borderColor: dataset.borderColor || CONFIG.CHART_COLORS.primary,
                backgroundColor: dataset.backgroundColor || Utils.hexToRgba(CONFIG.CHART_COLORS.primary, 0.1),
                tension: dataset.tension || 0.4,
                borderWidth: dataset.borderWidth || 2,
                pointRadius: dataset.pointRadius || 0,
                pointHoverRadius: dataset.pointHoverRadius || 6
            }))
        };
    }
};

// Event Bus for component communication
class EventBus {
    constructor() {
        this.events = {};
    }

    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }

    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }
}

const eventBus = new EventBus();

// Data Cache Management
class DataCache {
    constructor(maxSize = 100, ttl = 5 * 60 * 1000) { // 5 minutes TTL
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        const isExpired = Date.now() - item.timestamp > this.ttl;
        if (isExpired) {
            this.cache.delete(key);
            return null;
        }

        return item.value;
    }

    has(key) {
        return this.get(key) !== null;
    }

    clear() {
        this.cache.clear();
    }
}

const dataCache = new DataCache();

// Performance Monitor
class PerformanceMonitor {
    constructor() {
        this.marks = {};
        this.measures = {};
    }

    mark(name) {
        this.marks[name] = performance.now();
    }

    measure(name, startMark, endMark = null) {
        const startTime = this.marks[startMark];
        const endTime = endMark ? this.marks[endMark] : performance.now();
        
        if (startTime === undefined) {
            console.warn(`Start mark "${startMark}" not found`);
            return;
        }

        const duration = endTime - startTime;
        this.measures[name] = duration;
        
        console.log(`${name}: ${duration.toFixed(2)}ms`);
        return duration;
    }

    getMeasure(name) {
        return this.measures[name];
    }

    clearMarks() {
        this.marks = {};
    }

    clearMeasures() {
        this.measures = {};
    }
}

const perfMonitor = new PerformanceMonitor();

// Error Handler
class ErrorHandler {
    constructor() {
        this.setupGlobalHandlers();
    }

    setupGlobalHandlers() {
        window.addEventListener('error', (event) => {
            this.handleError(event.error, 'JavaScript Error');
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason, 'Unhandled Promise Rejection');
        });
    }

    handleError(error, type = 'Error') {
        console.error(`${type}:`, error);
        
        // Log to external service in production
        if (window.location.hostname !== 'localhost') {
            this.logToService(error, type);
        }

        // Show user-friendly message
        showToast('An unexpected error occurred. Please try refreshing the page.', 'error');
    }

    logToService(error, type) {
        // Implement external logging service integration
        // e.g., Sentry, LogRocket, etc.
        console.log('Would log to external service:', { error, type });
    }
}

const errorHandler = new ErrorHandler();

// Keyboard Shortcuts
class KeyboardShortcuts {
    constructor() {
        this.shortcuts = new Map();
        this.setupEventListeners();
    }

    register(keys, callback, description = '') {
        const normalizedKeys = this.normalizeKeys(keys);
        this.shortcuts.set(normalizedKeys, { callback, description });
    }

    unregister(keys) {
        const normalizedKeys = this.normalizeKeys(keys);
        this.shortcuts.delete(normalizedKeys);
    }

    normalizeKeys(keys) {
        return keys.toLowerCase()
            .replace(/\s+/g, '')
            .split('+')
            .sort()
            .join('+');
    }

    setupEventListeners() {
        document.addEventListener('keydown', (event) => {
            const pressedKeys = [];
            
            if (event.ctrlKey) pressedKeys.push('ctrl');
            if (event.shiftKey) pressedKeys.push('shift');
            if (event.altKey) pressedKeys.push('alt');
            if (event.metaKey) pressedKeys.push('meta');
            
            pressedKeys.push(event.key.toLowerCase());
            
            const keyCombo = pressedKeys.sort().join('+');
            const shortcut = this.shortcuts.get(keyCombo);
            
            if (shortcut) {
                event.preventDefault();
                shortcut.callback(event);
            }
        });
    }

    getRegisteredShortcuts() {
        const shortcuts = [];
        this.shortcuts.forEach((value, key) => {
            shortcuts.push({
                keys: key,
                description: value.description
            });
        });
        return shortcuts;
    }
}

const keyboardShortcuts = new KeyboardShortcuts();

// Initialize keyboard shortcuts
keyboardShortcuts.register('ctrl+k', () => {
    // Quick search functionality
    showToast('Quick search coming soon!', 'info');
}, 'Quick search');

keyboardShortcuts.register('ctrl+/', () => {
    // Help modal
    const helpModal = new Modal({
        title: 'Keyboard Shortcuts',
        content: createHelpContent(),
        size: 'lg'
    });
    helpModal.open();
}, 'Show help');

function createHelpContent() {
    const shortcuts = keyboardShortcuts.getRegisteredShortcuts();
    return `
        <div class="space-y-4">
            <p class="text-gray-600 dark:text-gray-400">Available keyboard shortcuts:</p>
            <div class="space-y-2">
                ${shortcuts.map(shortcut => `
                    <div class="flex justify-between items-center">
                        <span class="font-mono text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                            ${shortcut.keys.replace(/\+/g, ' + ').toUpperCase()}
                        </span>
                        <span class="text-sm text-gray-600 dark:text-gray-400">
                            ${shortcut.description}
                        </span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// File Upload Handler
class FileUploadHandler {
    constructor(element, options = {}) {
        this.element = typeof element === 'string' ? document.querySelector(element) : element;
        this.options = {
            maxFileSize: 10 * 1024 * 1024, // 10MB
            allowedTypes: ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
            multiple: false,
            ...options
        };
        this.files = [];
        this.setupEventListeners();
    }

    setupEventListeners() {
        if (!this.element) return;

        this.element.addEventListener('dragover', this.handleDragOver.bind(this));
        this.element.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.element.addEventListener('drop', this.handleDrop.bind(this));
        this.element.addEventListener('click', this.handleClick.bind(this));

        // Create hidden file input
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.multiple = this.options.multiple;
        this.fileInput.accept = this.options.allowedTypes.join(',');
        this.fileInput.style.display = 'none';
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        document.body.appendChild(this.fileInput);
    }

    handleDragOver(event) {
        event.preventDefault();
        this.element.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.element.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        this.element.classList.remove('dragover');
        
        const files = Array.from(event.dataTransfer.files);
        this.processFiles(files);
    }

    handleClick() {
        this.fileInput.click();
    }

    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        this.processFiles(files);
    }

    processFiles(files) {
        const validFiles = [];
        
        files.forEach(file => {
            if (this.validateFile(file)) {
                validFiles.push(file);
            }
        });

        if (validFiles.length > 0) {
            this.files = this.options.multiple ? [...this.files, ...validFiles] : [validFiles[0]];
            this.onFilesSelected(validFiles);
        }
    }

    validateFile(file) {
        // Check file size
        if (file.size > this.options.maxFileSize) {
            showToast(`File "${file.name}" is too large. Maximum size is ${this.formatFileSize(this.options.maxFileSize)}.`, 'error');
            return false;
        }

        // Check file type
        if (this.options.allowedTypes.length > 0 && !this.options.allowedTypes.includes(file.type)) {
            showToast(`File "${file.name}" has an unsupported format.`, 'error');
            return false;
        }

        return true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    onFilesSelected(files) {
        // Override this method or listen to custom event
        eventBus.emit('filesSelected', files);
    }

    clear() {
        this.files = [];
        this.fileInput.value = '';
    }
}

// Export global objects for use in other scripts
window.AppState = AppState;
window.api = api;
window.Utils = Utils;
window.Modal = Modal;
window.FormValidator = FormValidator;
window.ChartUtils = ChartUtils;
window.eventBus = eventBus;
window.dataCache = dataCache;
window.perfMonitor = perfMonitor;
window.keyboardShortcuts = keyboardShortcuts;
window.FileUploadHandler = FileUploadHandler;

// Initialize theme on load
document.addEventListener('DOMContentLoaded', function() {
    // Apply saved theme
    if (AppState.theme === 'dark') {
        document.documentElement.classList.add('dark');
    }

    // Initialize performance monitoring
    perfMonitor.mark('app-start');

    console.log('Advanced Sales Forecasting Platform loaded successfully');
});