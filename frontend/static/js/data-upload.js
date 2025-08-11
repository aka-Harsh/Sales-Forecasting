/**
 * Data Upload and Management JavaScript
 */

// File upload state management
const DataUploadState = {
    currentFile: null,
    uploadProgress: 0,
    validationResults: null,
    cleanedData: null,
    isProcessing: false
};

// Enhanced File Upload Handler with more features
class AdvancedFileUploadHandler extends FileUploadHandler {
    constructor(element, options = {}) {
        super(element, options);
        this.uploadQueue = [];
        this.currentUpload = null;
        this.retryAttempts = 3;
        
        // Additional options
        this.options = {
            ...this.options,
            autoProcess: true,
            showPreview: true,
            validateOnUpload: true,
            chunkSize: 1024 * 1024, // 1MB chunks for large files
            ...options
        };
        
        this.setupAdvancedFeatures();
    }
    
    setupAdvancedFeatures() {
        // Add file type icons to the drop zone
        this.updateDropZoneUI();
        
        // Setup progress tracking
        this.setupProgressTracking();
    }
    
    updateDropZoneUI() {
        const dropZone = this.element;
        const iconContainer = dropZone.querySelector('.file-type-icons');
        
        if (!iconContainer) {
            const iconsHtml = `
                <div class="file-type-icons flex justify-center space-x-4 text-xs text-gray-400 mt-4">
                    <span class="flex items-center"><i class="fas fa-file-csv mr-1 text-green-500"></i> CSV</span>
                    <span class="flex items-center"><i class="fas fa-file-excel mr-1 text-green-600"></i> Excel</span>
                    <span class="flex items-center"><i class="fas fa-file-code mr-1 text-blue-500"></i> JSON</span>
                </div>
            `;
            dropZone.insertAdjacentHTML('beforeend', iconsHtml);
        }
    }
    
    setupProgressTracking() {
        // Create progress elements if they don't exist
        if (!document.getElementById('upload-progress')) {
            const progressHtml = `
                <div id="upload-progress" class="hidden mb-6">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Uploading...</span>
                        <span id="progress-percentage" class="text-sm text-gray-500 dark:text-gray-400">0%</span>
                    </div>
                    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div id="progress-bar" class="bg-primary-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                    <div id="upload-details" class="mt-2 text-xs text-gray-500 dark:text-gray-400"></div>
                </div>
            `;
            this.element.insertAdjacentHTML('afterend', progressHtml);
        }
    }
    
    async processFiles(files) {
        if (!this.options.autoProcess) {
            this.uploadQueue.push(...files);
            return;
        }
        
        for (const file of files) {
            await this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        try {
            DataUploadState.currentFile = file;
            DataUploadState.isProcessing = true;
            
            // Show upload progress
            this.showProgress();
            
            // Update upload details
            this.updateUploadDetails(file);
            
            // Simulate chunked upload for large files
            if (file.size > this.options.chunkSize) {
                await this.uploadFileInChunks(file);
            } else {
                await this.uploadFileDirectly(file);
            }
            
        } catch (error) {
            this.handleUploadError(error, file);
        } finally {
            DataUploadState.isProcessing = false;
            setTimeout(() => this.hideProgress(), 2000);
        }
    }
    
    async uploadFileDirectly(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // Simulate upload progress
        const progressInterval = this.simulateProgress();
        
        try {
            const response = await api.uploadFile('/data/upload', file);
            
            clearInterval(progressInterval);
            this.setProgress(100);
            
            if (response.success) {
                this.handleUploadSuccess(response, file);
            } else {
                throw new Error(response.error || 'Upload failed');
            }
            
        } catch (error) {
            clearInterval(progressInterval);
            throw error;
        }
    }
    
    async uploadFileInChunks(file) {
        const chunks = Math.ceil(file.size / this.options.chunkSize);
        
        for (let i = 0; i < chunks; i++) {
            const start = i * this.options.chunkSize;
            const end = Math.min(start + this.options.chunkSize, file.size);
            const chunk = file.slice(start, end);
            
            await this.uploadChunk(chunk, i, chunks, file);
            
            // Update progress
            const progress = ((i + 1) / chunks) * 100;
            this.setProgress(progress);
        }
    }
    
    async uploadChunk(chunk, index, total, originalFile) {
        const formData = new FormData();
        formData.append('chunk', chunk);
        formData.append('chunkIndex', index);
        formData.append('totalChunks', total);
        formData.append('fileName', originalFile.name);
        formData.append('fileSize', originalFile.size);
        
        const response = await fetch('/api/data/upload-chunk', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Chunk upload failed: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    simulateProgress() {
        let progress = 0;
        return setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            this.setProgress(progress);
        }, 200);
    }
    
    showProgress() {
        const progressContainer = document.getElementById('upload-progress');
        if (progressContainer) {
            progressContainer.classList.remove('hidden');
        }
    }
    
    hideProgress() {
        const progressContainer = document.getElementById('upload-progress');
        if (progressContainer) {
            progressContainer.classList.add('hidden');
        }
    }
    
    setProgress(percentage) {
        DataUploadState.uploadProgress = percentage;
        
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = percentage + '%';
        }
        
        if (progressText) {
            progressText.textContent = Math.round(percentage) + '%';
        }
    }
    
    updateUploadDetails(file) {
        const detailsElement = document.getElementById('upload-details');
        if (detailsElement) {
            detailsElement.textContent = `Uploading ${file.name} (${this.formatFileSize(file.size)})`;
        }
    }
    
    handleUploadSuccess(response, file) {
        DataUploadState.uploadProgress = 100;
        
        // Show success message
        showToast(`${file.name} uploaded successfully!`, 'success');
        
        // Trigger file processed event
        eventBus.emit('fileUploaded', { response, file });
        
        // Auto-validate if option is enabled
        if (this.options.validateOnUpload) {
            setTimeout(() => {
                this.validateUploadedFile(response);
            }, 500);
        }
    }
    
    handleUploadError(error, file) {
        console.error('Upload error:', error);
        
        showToast(`Upload failed for ${file.name}: ${error.message}`, 'error');
        
        // Retry logic
        if (this.retryAttempts > 0) {
            this.retryAttempts--;
            showToast(`Retrying upload... (${this.retryAttempts} attempts left)`, 'info');
            
            setTimeout(() => {
                this.uploadFile(file);
            }, 2000);
        }
    }
    
    async validateUploadedFile(uploadResponse) {
        try {
            const validationResponse = await api.post('/data/validate', {
                filepath: uploadResponse.filepath
            });
            
            if (validationResponse.success) {
                DataUploadState.validationResults = validationResponse.validation;
                eventBus.emit('fileValidated', validationResponse);
                
                if (validationResponse.validation.is_valid) {
                    showToast('File validation passed!', 'success');
                } else {
                    showToast('File validation found issues', 'warning');
                }
            }
            
        } catch (error) {
            console.error('Validation error:', error);
            showToast('File validation failed', 'error');
        }
    }
    
    // Queue management
    processQueue() {
        if (this.uploadQueue.length === 0 || this.currentUpload) {
            return;
        }
        
        const file = this.uploadQueue.shift();
        this.uploadFile(file);
    }
    
    clearQueue() {
        this.uploadQueue = [];
    }
    
    // File preview functionality
    generateFilePreview(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                try {
                    let preview = '';
                    
                    if (file.type === 'text/csv') {
                        preview = DataUploadUtils.parseCSVPreview(e.target.result);
                    } else if (file.type.includes('json')) {
                        preview = DataUploadUtils.parseJSONPreview(e.target.result);
                    } else {
                        preview = '<p>Binary file preview not available</p>';
                    }
                    
                    resolve(preview);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = reject;
            
            if (file.type === 'text/csv' || file.type.includes('json')) {
                reader.readAsText(file);
            } else {
                resolve('<p>Preview not available for this file type</p>');
            }
        });
    }
}

// Data Upload Utilities
const DataUploadUtils = {
    // Parse CSV for preview
    parseCSVPreview(csvText, maxRows = 5) {
        const lines = csvText.split('\n').slice(0, maxRows + 1);
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        
        let html = '<div class="overflow-x-auto"><table class="min-w-full text-xs">';
        html += '<thead class="bg-gray-50 dark:bg-gray-700"><tr>';
        
        headers.forEach(header => {
            html += `<th class="px-2 py-1 text-left font-medium text-gray-700 dark:text-gray-300">${header}</th>`;
        });
        
        html += '</tr></thead><tbody>';
        
        for (let i = 1; i < lines.length && i <= maxRows; i++) {
            if (lines[i].trim()) {
                const cells = lines[i].split(',').map(c => c.trim().replace(/"/g, ''));
                html += '<tr class="border-t border-gray-200 dark:border-gray-600">';
                
                cells.forEach(cell => {
                    html += `<td class="px-2 py-1 text-gray-600 dark:text-gray-400">${cell}</td>`;
                });
                
                html += '</tr>';
            }
        }
        
        html += '</tbody></table></div>';
        return html;
    },

    // Parse JSON for preview
    parseJSONPreview(jsonText) {
        try {
            const data = JSON.parse(jsonText);
            const formatted = JSON.stringify(data, null, 2);
            
            // Limit preview to first 1000 characters
            const preview = formatted.length > 1000 ? formatted.substring(0, 1000) + '...' : formatted;
            
            return `<pre class="bg-gray-100 dark:bg-gray-800 p-3 rounded text-xs overflow-x-auto"><code>${this.escapeHtml(preview)}</code></pre>`;
        } catch (error) {
            return `<div class="text-red-600 dark:text-red-400">Invalid JSON format: ${error.message}</div>`;
        }
    },

    // Escape HTML to prevent XSS
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Validate file before upload
    validateFile(file, options = {}) {
        const errors = [];
        const warnings = [];
        
        // File size check
        const maxSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            errors.push(`File size (${this.formatFileSize(file.size)}) exceeds maximum allowed size (${this.formatFileSize(maxSize)})`);
        }
        
        // File type check
        const allowedTypes = options.allowedTypes || ['text/csv', 'application/json', 'application/vnd.ms-excel'];
        if (!allowedTypes.includes(file.type)) {
            errors.push(`File type "${file.type}" is not supported`);
        }
        
        // File name validation
        if (file.name.length > 255) {
            errors.push('File name is too long (maximum 255 characters)');
        }
        
        // CSV specific validation
        if (file.type === 'text/csv') {
            if (!file.name.toLowerCase().endsWith('.csv')) {
                warnings.push('File extension does not match CSV type');
            }
        }
        
        return { errors, warnings, isValid: errors.length === 0 };
    },

    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Get file type icon
    getFileTypeIcon(file) {
        const type = file.type.toLowerCase();
        
        if (type.includes('csv')) {
            return '<i class="fas fa-file-csv text-green-500"></i>';
        } else if (type.includes('excel') || type.includes('spreadsheet')) {
            return '<i class="fas fa-file-excel text-green-600"></i>';
        } else if (type.includes('json')) {
            return '<i class="fas fa-file-code text-blue-500"></i>';
        } else {
            return '<i class="fas fa-file text-gray-500"></i>';
        }
    },

    // Generate data quality report
    generateQualityReport(data) {
        const report = {
            totalRecords: data.length,
            missingValues: 0,
            duplicateRecords: 0,
            dataTypes: {},
            valueRanges: {},
            qualityScore: 100
        };
        
        if (data.length === 0) {
            report.qualityScore = 0;
            return report;
        }
        
        const columns = Object.keys(data[0]);
        
        columns.forEach(column => {
            const columnData = data.map(row => row[column]).filter(val => val !== null && val !== undefined && val !== '');
            
            // Missing values
            const missingCount = data.length - columnData.length;
            report.missingValues += missingCount;
            
            // Data types
            const types = new Set(columnData.map(val => typeof val));
            report.dataTypes[column] = Array.from(types);
            
            // Value ranges for numeric columns
            const numericValues = columnData.filter(val => !isNaN(val) && val !== '').map(Number);
            if (numericValues.length > 0) {
                report.valueRanges[column] = {
                    min: Math.min(...numericValues),
                    max: Math.max(...numericValues),
                    mean: numericValues.reduce((a, b) => a + b, 0) / numericValues.length
                };
            }
        });
        
        // Calculate quality score
        const missingValuesPenalty = (report.missingValues / (data.length * columns.length)) * 50;
        report.qualityScore = Math.max(0, 100 - missingValuesPenalty);
        
        return report;
    }
};

// Data Processing Pipeline
class DataProcessingPipeline {
    constructor() {
        this.steps = [];
        this.results = [];
    }
    
    addStep(stepFunction, stepName) {
        this.steps.push({ function: stepFunction, name: stepName });
        return this;
    }
    
    async execute(data) {
        this.results = [];
        let currentData = data;
        
        for (const step of this.steps) {
            try {
                console.log(`Executing step: ${step.name}`);
                const result = await step.function(currentData);
                
                this.results.push({
                    stepName: step.name,
                    success: true,
                    data: result.data || result,
                    metadata: result.metadata || {}
                });
                
                currentData = result.data || result;
                
            } catch (error) {
                console.error(`Step ${step.name} failed:`, error);
                
                this.results.push({
                    stepName: step.name,
                    success: false,
                    error: error.message,
                    data: currentData
                });
                
                break; // Stop pipeline on error
            }
        }
        
        return {
            finalData: currentData,
            results: this.results,
            success: this.results.every(r => r.success)
        };
    }
    
    getStepResult(stepName) {
        return this.results.find(r => r.stepName === stepName);
    }
}

// Data transformation utilities
const DataTransforms = {
    // Clean column names
    cleanColumnNames(data) {
        if (!Array.isArray(data) || data.length === 0) return data;
        
        const cleanedData = data.map(row => {
            const cleanedRow = {};
            Object.keys(row).forEach(key => {
                const cleanKey = key.trim()
                    .toLowerCase()
                    .replace(/\s+/g, '_')
                    .replace(/[^a-z0-9_]/g, '');
                cleanedRow[cleanKey] = row[key];
            });
            return cleanedRow;
        });
        
        return {
            data: cleanedData,
            metadata: { originalColumns: Object.keys(data[0]), cleanedColumns: Object.keys(cleanedData[0]) }
        };
    },
    
    // Remove empty rows
    removeEmptyRows(data) {
        if (!Array.isArray(data)) return data;
        
        const originalCount = data.length;
        const cleanedData = data.filter(row => {
            return Object.values(row).some(value => 
                value !== null && value !== undefined && value !== ''
            );
        });
        
        return {
            data: cleanedData,
            metadata: { originalCount, cleanedCount: cleanedData.length, removedCount: originalCount - cleanedData.length }
        };
    },
    
    // Convert data types
    convertDataTypes(data, typeMap = {}) {
        if (!Array.isArray(data) || data.length === 0) return data;
        
        const convertedData = data.map(row => {
            const convertedRow = { ...row };
            
            Object.keys(typeMap).forEach(column => {
                if (convertedRow.hasOwnProperty(column)) {
                    const targetType = typeMap[column];
                    const value = convertedRow[column];
                    
                    try {
                        switch (targetType) {
                            case 'number':
                                convertedRow[column] = value === '' ? null : Number(value);
                                break;
                            case 'date':
                                convertedRow[column] = value === '' ? null : new Date(value);
                                break;
                            case 'boolean':
                                convertedRow[column] = value === 'true' || value === '1' || value === 1;
                                break;
                            case 'string':
                                convertedRow[column] = String(value);
                                break;
                        }
                    } catch (error) {
                        console.warn(`Failed to convert ${column} to ${targetType}:`, error);
                    }
                }
            });
            
            return convertedRow;
        });
        
        return {
            data: convertedData,
            metadata: { typeMap, conversionsApplied: Object.keys(typeMap) }
        };
    },
    
    // Detect and handle outliers
    detectOutliers(data, columns = [], method = 'iqr') {
        if (!Array.isArray(data) || data.length === 0) return data;
        
        const outliers = [];
        const cleanedData = [...data];
        
        columns.forEach(column => {
            const values = data.map(row => row[column]).filter(val => !isNaN(val) && val !== null).map(Number);
            
            if (values.length === 0) return;
            
            let outlierIndices = [];
            
            if (method === 'iqr') {
                values.sort((a, b) => a - b);
                const q1 = values[Math.floor(values.length * 0.25)];
                const q3 = values[Math.floor(values.length * 0.75)];
                const iqr = q3 - q1;
                const lowerBound = q1 - 1.5 * iqr;
                const upperBound = q3 + 1.5 * iqr;
                
                data.forEach((row, index) => {
                    const value = Number(row[column]);
                    if (!isNaN(value) && (value < lowerBound || value > upperBound)) {
                        outlierIndices.push(index);
                        outliers.push({ index, column, value, method });
                    }
                });
            }
        });
        
        return {
            data: cleanedData,
            metadata: { outliers, method, columnsAnalyzed: columns }
        };
    }
};

// Export utilities for global use
window.AdvancedFileUploadHandler = AdvancedFileUploadHandler;
window.DataUploadUtils = DataUploadUtils;
window.DataProcessingPipeline = DataProcessingPipeline;
window.DataTransforms = DataTransforms;
window.DataUploadState = DataUploadState;

// Initialize data upload functionality when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Setup event listeners for data upload events
    eventBus.on('fileUploaded', handleFileUploaded);
    eventBus.on('fileValidated', handleFileValidated);
    eventBus.on('dataProcessed', handleDataProcessed);
});

function handleFileUploaded(data) {
    console.log('File uploaded:', data);
    
    // Update UI to show upload success
    const uploadStatus = document.getElementById('upload-status');
    if (uploadStatus) {
        uploadStatus.className = 'alert success';
        uploadStatus.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-check-circle mr-3"></i>
                <div>
                    <p class="font-medium">Upload successful!</p>
                    <p class="text-sm">File: ${data.file.name}</p>
                </div>
            </div>
        `;
        uploadStatus.classList.remove('hidden');
    }
    
    // Show data preview section
    const previewSection = document.getElementById('data-preview-section');
    if (previewSection) {
        previewSection.classList.remove('hidden');
    }
}

function handleFileValidated(data) {
    console.log('File validated:', data);
    
    DataUploadState.validationResults = data.validation;
    
    // Update validation UI
    const validationSection = document.getElementById('validation-results');
    if (validationSection) {
        validationSection.classList.remove('hidden');
        displayValidationResults(data.validation);
    }
}

function handleDataProcessed(data) {
    console.log('Data processed:', data);
    
    DataUploadState.cleanedData = data;
    
    // Update data visualization
    const visualizationSection = document.getElementById('data-visualization');
    if (visualizationSection) {
        visualizationSection.classList.remove('hidden');
    }
}

function displayValidationResults(validation) {
    const validationContent = document.getElementById('validation-content');
    if (!validationContent) return;
    
    let html = '';
    
    if (validation.is_valid) {
        html += `
            <div class="alert success mb-4">
                <div class="flex items-center">
                    <i class="fas fa-check-circle mr-3"></i>
                    <span class="font-medium">Data validation passed!</span>
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="alert danger mb-4">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-circle mr-3"></i>
                    <span class="font-medium">Data validation failed</span>
                </div>
            </div>
        `;
    }
    
    // Display errors
    if (validation.errors && validation.errors.length > 0) {
        html += `
            <div class="mb-4">
                <h4 class="font-medium text-red-800 dark:text-red-200 mb-2">Errors:</h4>
                <ul class="list-disc list-inside space-y-1">
                    ${validation.errors.map(error => `<li class="text-red-700 dark:text-red-300">${error}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Display warnings
    if (validation.warnings && validation.warnings.length > 0) {
        html += `
            <div class="mb-4">
                <h4 class="font-medium text-yellow-800 dark:text-yellow-200 mb-2">Warnings:</h4>
                <ul class="list-disc list-inside space-y-1">
                    ${validation.warnings.map(warning => `<li class="text-yellow-700 dark:text-yellow-300">${warning}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Display validation info
    if (validation.info) {
        html += `
            <div class="mb-4">
                <h4 class="font-medium text-blue-800 dark:text-blue-200 mb-2">Data Information:</h4>
                <div class="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                    <p class="text-sm text-blue-700 dark:text-blue-300">
                        <strong>Date Column:</strong> ${validation.info.date_column || 'Not detected'}<br>
                        <strong>Sales Column:</strong> ${validation.info.sales_column || 'Not detected'}<br>
                        <strong>Total Records:</strong> ${validation.info.total_records || 'Unknown'}
                    </p>
                </div>
            </div>
        `;
    }
    
    validationContent.innerHTML = html;
}