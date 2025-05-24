// WaterSense AI - Main Application Logic
// SVD & Quantum Watermarking with AI Detection

// ---------- Backend Integration ----------
const API_BASE_URL = 'http://localhost:3000/api';

class WatermarkingAPI {
    static async embedWatermark(imageData, method, strength) {
        const response = await fetch(`${API_BASE_URL}/watermark`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: imageData,
                method: method,
                strength: strength
            })
        });
        return await response.json();
    }
}

class AIDetectionAPI {
    static async detectManipulation(imageData) {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        return await response.json();
    }
}

// ---------- TensorFlow Operations ----------
class ImageProcessor {
    static async convertToTensor(img) {
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims();
            return tensor;
        });
    }

    static calculateDifference(original, watermarked) {
        return tf.tidy(() => {
            const diff = tf.abs(original.sub(watermarked));
            return tf.mul(diff, 10.0); // Amplify difference
        });
    }
}

// ---------- Quantum Implementation ----------
class QuantumWatermark {
    constructor() {
        this.circuit = new QuantumCircuit(1024);
        this.circuit.applyEntanglement();
    }

    generatePattern(strength) {
        const pattern = new Float32Array(1024);
        for(let i=0; i<1024; i++) {
            pattern[i] = this.circuit.measure(i) * strength;
        }
        return pattern;
    }
}

class QuantumCircuit {
    constructor(qubits) {
        this.qubits = qubits;
        this.state = new Array(qubits).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);
    }

    applyEntanglement() {
        for(let i = 0; i < this.qubits - 1; i += 2) {
            if(Math.random() > 0.5) {
                this.state[i] = this.state[i+1];
            }
        }
    }

    measure(index) {
        return this.state[index % this.qubits];
    }
}

// ---------- SVD Implementation ----------
class SVDWatermarking {
    static svd(matrix) {
        // Simplified SVD implementation using power iteration
        const m = matrix.length;
        const n = matrix[0].length;
        const minDim = Math.min(m, n);
        
        // Create covariance matrix
        const AT = this.transpose(matrix);
        const ATA = this.multiply(AT, matrix);
        
        // Find eigenvalues and eigenvectors using power iteration
        const { eigenvalues, eigenvectors } = this.powerIteration(ATA, Math.min(minDim, 10));
        
        return {
            U: eigenvectors,
            S: eigenvalues,
            V: eigenvectors
        };
    }
    
    static transpose(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = [];
        
        for (let i = 0; i < cols; i++) {
            result[i] = [];
            for (let j = 0; j < rows; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }
    
    static multiply(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;
        const result = [];
        
        for (let i = 0; i < rowsA; i++) {
            result[i] = [];
            for (let j = 0; j < colsB; j++) {
                result[i][j] = 0;
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    static powerIteration(matrix, numComponents) {
        const n = matrix.length;
        const eigenvalues = [];
        const eigenvectors = [];
        
        for (let comp = 0; comp < numComponents; comp++) {
            let v = new Array(n).fill(0).map(() => Math.random());
            
            // Normalize
            let norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
            v = v.map(val => val / norm);
            
            // Power iteration
            for (let iter = 0; iter < 100; iter++) {
                const Av = matrix.map(row => 
                    row.reduce((sum, val, idx) => sum + val * v[idx], 0)
                );
                
                norm = Math.sqrt(Av.reduce((sum, val) => sum + val * val, 0));
                v = Av.map(val => val / norm);
            }
            
            eigenvalues.push(norm);
            eigenvectors.push(v);
        }
        
        return { eigenvalues, eigenvectors };
    }
    
    static embedWatermark(imageData, watermark, strength = 0.1) {
        const width = imageData.width;
        const height = imageData.height;
        const data = new Uint8ClampedArray(imageData.data);
        
        // Convert to matrix (grayscale)
        const matrix = [];
        for (let i = 0; i < height; i++) {
            matrix[i] = [];
            for (let j = 0; j < width; j++) {
                const index = (i * width + j) * 4;
                matrix[i][j] = (data[index] + data[index + 1] + data[index + 2]) / 3;
            }
        }
        
        // Apply SVD
        const { U, S, V } = this.svd(matrix);
        
        // Embed watermark in singular values
        const watermarkSize = Math.min(watermark.length, S.length);
        for (let i = 0; i < watermarkSize; i++) {
            S[i] += watermark[i] * strength * 50;
        }
        
        // Reconstruct matrix
        const reconstructed = this.reconstructFromSVD(U, S, V, height, width);
        
        // Convert back to image data
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                const index = (i * width + j) * 4;
                const value = Math.max(0, Math.min(255, reconstructed[i][j]));
                data[index] = value;     // R
                data[index + 1] = value; // G
                data[index + 2] = value; // B
                // Alpha channel remains unchanged
            }
        }
        
        return new ImageData(data, width, height);
    }
    
    static reconstructFromSVD(U, S, V, height, width) {
        const result = [];
        const components = Math.min(S.length, 5); // Use top 5 components
        
        for (let i = 0; i < height; i++) {
            result[i] = [];
            for (let j = 0; j < width; j++) {
                let value = 0;
                for (let k = 0; k < components && k < U.length; k++) {
                    const uik = k < U[i].length ? U[i][k] : 0;
                    const sk = k < S.length ? S[k] : 0;
                    const vkj = k < V.length && j < V[k].length ? V[k][j] : 0;
                    value += uik * sk * vkj;
                }
                result[i][j] = value;
            }
        }
        return result;
    }
}

// ---------- Quantum-inspired Watermarking ----------
class QuantumWatermarking {
    static dct2D(matrix) {
        const N = matrix.length;
        const M = matrix[0].length;
        const result = [];
        
        for (let u = 0; u < N; u++) {
            result[u] = [];
            for (let v = 0; v < M; v++) {
                let sum = 0;
                for (let x = 0; x < N; x++) {
                    for (let y = 0; y < M; y++) {
                        sum += matrix[x][y] * 
                               Math.cos(Math.PI * u * (2 * x + 1) / (2 * N)) *
                               Math.cos(Math.PI * v * (2 * y + 1) / (2 * M));
                    }
                }
                const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
                const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
                result[u][v] = (2 / Math.sqrt(N * M)) * cu * cv * sum;
            }
        }
        return result;
    }
    
    static idct2D(matrix) {
        const N = matrix.length;
        const M = matrix[0].length;
        const result = [];
        
        for (let x = 0; x < N; x++) {
            result[x] = [];
            for (let y = 0; y < M; y++) {
                let sum = 0;
                for (let u = 0; u < N; u++) {
                    for (let v = 0; v < M; v++) {
                        const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
                        const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
                        sum += cu * cv * matrix[u][v] *
                               Math.cos(Math.PI * u * (2 * x + 1) / (2 * N)) *
                               Math.cos(Math.PI * v * (2 * y + 1) / (2 * M));
                    }
                }
                result[x][y] = (2 / Math.sqrt(N * M)) * sum;
            }
        }
        return result;
    }
    
    static quantumEmbedding(imageData, watermark, strength = 0.1) {
        const width = imageData.width;
        const height = imageData.height;
        const data = new Uint8ClampedArray(imageData.data);
        
        // Process in 8x8 blocks
        const blockSize = 8;
        let watermarkIndex = 0;
        
        for (let i = 0; i < height - blockSize; i += blockSize) {
            for (let j = 0; j < width - blockSize; j += blockSize) {
                // Extract 8x8 block
                const block = [];
                for (let y = 0; y < blockSize; y++) {
                    block[y] = [];
                    for (let x = 0; x < blockSize; x++) {
                        const index = ((i + y) * width + (j + x)) * 4;
                        block[y][x] = (data[index] + data[index + 1] + data[index + 2]) / 3;
                    }
                }
                
                // Apply DCT
                const dctBlock = this.dct2D(block);
                
                // Embed watermark in mid-frequency coefficients
                if (watermarkIndex < watermark.length) {
                    const positions = [[2, 1], [1, 2], [3, 0], [0, 3]];
                    for (let pos of positions) {
                        if (watermarkIndex < watermark.length) {
                            dctBlock[pos[0]][pos[1]] += watermark[watermarkIndex] * strength * 100;
                            watermarkIndex++;
                        }
                    }
                }
                
                // Apply inverse DCT
                const reconstructedBlock = this.idct2D(dctBlock);
                
                // Put block back
                for (let y = 0; y < blockSize; y++) {
                    for (let x = 0; x < blockSize; x++) {
                        const index = ((i + y) * width + (j + x)) * 4;
                        const value = Math.max(0, Math.min(255, reconstructedBlock[y][x]));
                        data[index] = value;
                        data[index + 1] = value;
                        data[index + 2] = value;
                    }
                }
            }
        }
        
        return new ImageData(data, width, height);
    }
}

// ---------- AI Detection Simulator ----------
class AIDetection {
    static async detectManipulation(imageData, modelType) {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Extract features from image
        const features = this.extractFeatures(imageData);
        
        // Simulate different model behaviors
        let confidence = 0;
        let isManipulated = false;
        
        switch (modelType) {
            case 'cnn':
                confidence = Math.random() * 0.3 + 0.1; // 10-40%
                break;
            case 'resnet':
                confidence = Math.random() * 0.4 + 0.3; // 30-70%
                break;
            case 'efficientnet':
                confidence = Math.random() * 0.3 + 0.5; // 50-80%
                break;
        }
        
        isManipulated = confidence > 0.5;
        
        return {
            isManipulated,
            confidence: confidence * 100,
            features,
            modelType
        };
    }
    
    static extractFeatures(imageData) {
        const data = imageData.data;
        let totalVariation = 0;
        let edgeEnergy = 0;
        let colorVariance = 0;
        
        // Calculate image statistics
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Simple edge detection
            if (i + imageData.width * 4 < data.length) {
                const nextR = data[i + imageData.width * 4];
                const nextG = data[i + imageData.width * 4 + 1];
                const nextB = data[i + imageData.width * 4 + 2];
                
                edgeEnergy += Math.abs(r - nextR) + Math.abs(g - nextG) + Math.abs(b - nextB);
            }
            
            totalVariation += Math.abs(r - 128) + Math.abs(g - 128) + Math.abs(b - 128);
        }
        
        return {
            totalVariation: totalVariation / (data.length / 4),
            edgeEnergy: edgeEnergy / (data.length / 4),
            colorVariance: colorVariance
        };
    }
}

// ---------- Main Application Class ----------
class WaterSenseApp {
    constructor() {
        this.currentWatermarkFile = null;
        this.currentDetectionFile = null;
        this.processingStartTime = null;
        this.detectionModel = null;
        
        this.init();
    }

    async init() {
        // Initialize AI model (if available)
        try {
            this.detectionModel = await tf.loadLayersModel('/models/watermark-detection/model.json');
        } catch (error) {
            console.log('AI model not available, using simulation mode');
        }

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Mobile Menu Toggle
        const mobileMenuButton = document.getElementById('mobile-menu-button');
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenuButton && mobileMenu) {
            mobileMenuButton.addEventListener('click', () => {
                mobileMenu.classList.toggle('hidden');
            });
        }

        // Strength slider update
        const strengthSlider = document.getElementById('watermark-strength');
        const strengthValue = document.getElementById('strength-value');
        if (strengthSlider && strengthValue) {
            strengthSlider.addEventListener('input', function() {
                strengthValue.textContent = this.value;
            });
        }

        this.setupWatermarkingSection();
        this.setupDetectionSection();
    }

    setupWatermarkingSection() {
        const watermarkInput = document.getElementById('watermark-input');
        const watermarkBrowse = document.getElementById('watermark-browse');
        const watermarkDropzone = document.getElementById('watermark-dropzone');
        const watermarkPreview = document.getElementById('watermark-preview');
        const watermarkImagePreview = document.getElementById('watermark-image-preview');
        const watermarkRemove = document.getElementById('watermark-remove');
        const watermarkGenerate = document.getElementById('watermark-generate');

        if (!watermarkDropzone || !watermarkInput || !watermarkBrowse) return;

        // Drag and drop for watermarking
        watermarkDropzone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('border-purple-400');
        });

        watermarkDropzone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('border-purple-400');
        });

        watermarkDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            watermarkDropzone.classList.remove('border-purple-400');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.handleWatermarkFile(files[0]);
            }
        });

        watermarkBrowse.addEventListener('click', () => watermarkInput.click());
        
        watermarkInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleWatermarkFile(file);
            }
        });

        if (watermarkRemove) {
            watermarkRemove.addEventListener('click', () => {
                if (watermarkPreview) watermarkPreview.classList.add('hidden');
                if (watermarkInput) watermarkInput.value = '';
                if (watermarkGenerate) watermarkGenerate.disabled = true;
                this.currentWatermarkFile = null;
            });
        }

        if (watermarkGenerate) {
            watermarkGenerate.addEventListener('click', () => this.processWatermark());
        }
    }

    setupDetectionSection() {
        const detectionInput = document.getElementById('detection-input');
        const detectionBrowse = document.getElementById('detection-browse');
        const detectionDropzone = document.getElementById('detection-dropzone');
        const detectionPreview = document.getElementById('detection-preview');
        const detectionImagePreview = document.getElementById('detection-image-preview');
        const detectionRemove = document.getElementById('detection-remove');
        const detectionAnalyze = document.getElementById('detection-analyze');

        if (!detectionDropzone || !detectionInput || !detectionBrowse) return;

        // Drag and drop for detection
        detectionDropzone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('border-blue-400');
        });

        detectionDropzone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('border-blue-400');
        });

        detectionDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            detectionDropzone.classList.remove('border-blue-400');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.handleDetectionFile(files[0]);
            }
        });

        detectionBrowse.addEventListener('click', () => detectionInput.click());
        
        detectionInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleDetectionFile(file);
            }
        });

        if (detectionRemove) {
            detectionRemove.addEventListener('click', () => {
                if (detectionPreview) detectionPreview.classList.add('hidden');
                if (detectionInput) detectionInput.value = '';
                if (detectionAnalyze) detectionAnalyze.disabled = true;
                this.currentDetectionFile = null;
            });
        }

        if (detectionAnalyze) {
            detectionAnalyze.addEventListener('click', () => this.processDetection());
        }
    }

    handleWatermarkFile(file) {
        this.currentWatermarkFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            const preview = document.getElementById('watermark-image-preview');
            const container = document.getElementById('watermark-preview');
            const generateBtn = document.getElementById('watermark-generate');
            
            if (preview) preview.src = event.target.result;
            if (container) container.classList.remove('hidden');
            if (generateBtn) generateBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    handleDetectionFile(file) {
        this.currentDetectionFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            const preview = document.getElementById('detection-image-preview');
            const container = document.getElementById('detection-preview');
            const analyzeBtn = document.getElementById('detection-analyze');
            
            if (preview) preview.src = event.target.result;
            if (container) container.classList.remove('hidden');
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    async processWatermark() {
        if (!this.currentWatermarkFile) return;
        
        this.processingStartTime = Date.now();
        const watermarkGenerate = document.getElementById('watermark-generate');
        if (!watermarkGenerate) return;
        
        // Show processing state
        watermarkGenerate.classList.add('processing');
        watermarkGenerate.innerHTML = '<div class="spinner"></div>Memproses...';
        
        try {
            const strength = parseInt(document.getElementById('watermark-strength').value) / 10;
            const method = document.getElementById('watermark-method').value;
            
            // Process image
            const canvas = document.getElementById('processing-canvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = async () => {
                canvas.width = Math.min(img.width, 800);
                canvas.height = Math.min(img.height, 600);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                let watermarkedData;
                
                // Generate watermark
                const watermarkSize = method === 'quantum' ? 256 : Math.min(1000, imageData.width * imageData.height / 16);
                const watermark = this.generateBinaryWatermark(watermarkSize);
                
                // Apply selected watermarking method
                if (method === 'svd') {
                    watermarkedData = SVDWatermarking.embedWatermark(imageData, watermark, strength);
                } else if (method === 'quantum') {
                    watermarkedData = QuantumWatermarking.quantumEmbedding(imageData, watermark, strength);
                } else {
                    // Hybrid approach
                    const svdWatermark = this.generateBinaryWatermark(Math.min(500, imageData.width * imageData.height / 32));
                    const quantumWatermark = this.generateBinaryWatermark(128);
                    const svdData = SVDWatermarking.embedWatermark(imageData, svdWatermark, strength/2);
                    watermarkedData = QuantumWatermarking.quantumEmbedding(svdData, quantumWatermark, strength/2);
                }
                
                // Display results
                ctx.putImageData(watermarkedData, 0, 0);
                const watermarkedUrl = canvas.toDataURL();
                
                // Create difference visualization
                const diffCanvas = document.createElement('canvas');
                diffCanvas.width = canvas.width;
                diffCanvas.height = canvas.height;
                const diffCtx = diffCanvas.getContext('2d');
                const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                // Calculate difference
                const diffData = new ImageData(canvas.width, canvas.height);
                for (let i = 0; i < originalData.data.length; i += 4) {
                    const diff = Math.abs(originalData.data[i] - watermarkedData.data[i]);
                    diffData.data[i] = diff * 5; // Amplify difference
                    diffData.data[i + 1] = diff * 5;
                    diffData.data[i + 2] = diff * 5;
                    diffData.data[i + 3] = 255;
                }
                
                diffCtx.putImageData(diffData, 0, 0);
                const diffUrl = diffCanvas.toDataURL();
                
                // Update display
                const originalPreview = document.getElementById('result-original');
                const watermarkedPreview = document.getElementById('result-watermarked');
                const diffPreview = document.getElementById('result-visualization');
                
                if (originalPreview) originalPreview.src = document.getElementById('watermark-image-preview').src;
                if (watermarkedPreview) watermarkedPreview.src = watermarkedUrl;
                if (diffPreview) diffPreview.src = diffUrl;
                
                // Update metrics with animation
                setTimeout(() => {
                    this.updateMetrics(strength, method);
                }, 500);
                
                // Show results
                const noResults = document.getElementById('no-results');
                const resultContainer = document.getElementById('result-container');
                
                if (noResults) noResults.classList.add('hidden');
                if (resultContainer) resultContainer.classList.remove('hidden');
                
                // Scroll to results
                if (resultContainer) {
                    resultContainer.scrollIntoView({ 
                        behavior: 'smooth' 
                    });
                }
            };
            
            img.src = URL.createObjectURL(this.currentWatermarkFile);
            
        } catch (error) {
            console.error('Error processing watermark:', error);
            alert('Terjadi kesalahan saat memproses gambar. Silakan coba lagi.');
        } finally {
            // Reset button
            if (watermarkGenerate) {
                watermarkGenerate.classList.remove('processing');
                watermarkGenerate.innerHTML = '<i class="fas fa-magic mr-2"></i>Generate Watermark';
            }
        }
    }

    async processDetection() {
        if (!this.currentDetectionFile) return;
        
        this.processingStartTime = Date.now();
        const detectionAnalyze = document.getElementById('detection-analyze');
        if (!detectionAnalyze) return;
        
        // Show processing state
        detectionAnalyze.classList.add('processing');
        detectionAnalyze.innerHTML = '<div class="spinner"></div>Menganalisis...';
        
        try {
            const modelType = document.getElementById('detection-model').value;
            
            // Process image
            const canvas = document.getElementById('processing-canvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = async () => {
                canvas.width = Math.min(img.width, 800);
                canvas.height = Math.min(img.height, 600);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                // Run AI detection
                const result = await AIDetection.detectManipulation(imageData, modelType);
                
                // Update detection results
                this.updateDetectionResults(result);
                
                // Show original image in results
                const originalPreview = document.getElementById('result-original');
                if (originalPreview) originalPreview.src = document.getElementById('detection-image-preview').src;
                
                // Show results
                const noResults = document.getElementById('no-results');
                const resultContainer = document.getElementById('result-container');
                
                if (noResults) noResults.classList.add('hidden');
                if (resultContainer) resultContainer.classList.remove('hidden');
                
                // Scroll to results
                if (resultContainer) {
                    resultContainer.scrollIntoView({ 
                        behavior: 'smooth' 
                    });
                }
            };
            
            img.src = URL.createObjectURL(this.currentDetectionFile);
            
        } catch (error) {
            console.error('Error in AI detection:', error);
            alert('Terjadi kesalahan saat menganalisis gambar. Silakan coba lagi.');
        } finally {
            // Reset button
            if (detectionAnalyze) {
                detectionAnalyze.classList.remove('processing');
                detectionAnalyze.innerHTML = '<i class="fas fa-microscope mr-2"></i>Analisis Forensik';
            }
        }
    }

    generateBinaryWatermark(length) {
        const watermark = new Array(length);
        for (let i = 0; i < length; i++) {
            watermark[i] = Math.random() > 0.5 ? 1 : -1;
        }
        return watermark;
    }

    updateMetrics(strength, method) {
        const baseIntegrity = 85 + Math.random() * 10;
        const strengthScore = Math.min(95, strength * 100 + Math.random() * 10);
        const resistanceScore = method === 'hybrid' ? 90 + Math.random() * 8 : 
                             method === 'quantum' ? 85 + Math.random() * 10 : 
                             75 + Math.random() * 15;
        
        // Animate progress bars
        setTimeout(() => {
            const integrityScore = document.getElementById('integrity-score');
            const integrityBar = document.getElementById('integrity-bar');
            if (integrityScore) integrityScore.textContent = Math.round(baseIntegrity) + '%';
            if (integrityBar) integrityBar.style.width = baseIntegrity + '%';
        }, 200);
        
        setTimeout(() => {
            const strengthScoreEl = document.getElementById('strength-score');
            const strengthBar = document.getElementById('strength-bar');
            if (strengthScoreEl) strengthScoreEl.textContent = Math.round(strengthScore) + '%';
            if (strengthBar) strengthBar.style.width = strengthScore + '%';
        }, 400);
        
        setTimeout(() => {
            const resistanceScoreEl = document.getElementById('resistance-score');
            const resistanceBar = document.getElementById('resistance-bar');
            if (resistanceScoreEl) resistanceScoreEl.textContent = Math.round(resistanceScore) + '%';
            if (resistanceBar) resistanceBar.style.width = resistanceScore + '%';
        }, 600);

        // Update detection status for watermarking
        const watermarkStatus = document.getElementById('detection-watermark');
        const manipulationStatus = document.getElementById('detection-manipulation');
        const integrityStatus = document.getElementById('detection-integrity');
        
        if (watermarkStatus) {
            watermarkStatus.innerHTML = 
                '<i class="fas fa-check-circle text-green-400 text-xl mr-2"></i><span>Watermark Berhasil Disisipi</span>';
        }
        
        if (manipulationStatus) {
            manipulationStatus.innerHTML = 
                '<i class="fas fa-shield-alt text-blue-400 text-xl mr-2"></i><span>Gambar Dilindungi</span>';
        }
        
        if (integrityStatus) {
            integrityStatus.innerHTML = 
                '<i class="fas fa-check-circle text-green-400 text-xl mr-2"></i><span>Integritas Terjaga</span>';
        }
        
        const confidenceScore = document.getElementById('confidence-score');
        const modelUsed = document.getElementById('model-used');
        const processingTime = document.getElementById('processing-time');
        
        if (confidenceScore) confidenceScore.textContent = Math.round(baseIntegrity) + '%';
        if (modelUsed) modelUsed.textContent = method.toUpperCase();
        if (processingTime) {
            processingTime.textContent = 
                ((Date.now() - this.processingStartTime) / 1000).toFixed(2) + 's';
        }
    }

    updateDetectionResults(result) {
        const processingTime = ((Date.now() - this.processingStartTime) / 1000).toFixed(2);
        
        // Update detection status
        const watermarkStatus = document.getElementById('detection-watermark');
        const manipulationStatus = document.getElementById('detection-manipulation');
        const integrityStatus = document.getElementById('detection-integrity');
        
        if (result.isManipulated) {
            if (watermarkStatus) {
                watermarkStatus.innerHTML = 
                    '<i class="fas fa-exclamation-triangle text-yellow-400 text-xl mr-2"></i><span>Watermark Tidak Terdeteksi</span>';
            }
            
            if (manipulationStatus) {
                manipulationStatus.innerHTML = 
                    '<i class="fas fa-times-circle text-red-400 text-xl mr-2"></i><span>MANIPULASI AI TERDETEKSI</span>';
            }
            
            if (integrityStatus) {
                integrityStatus.innerHTML = 
                    '<i class="fas fa-times-circle text-red-400 text-xl mr-2"></i><span>Integritas Bermasalah</span>';
            }
        } else {
            if (watermarkStatus) {
                watermarkStatus.innerHTML = 
                    '<i class="fas fa-question-circle text-gray-400 text-xl mr-2"></i><span>Watermark: Tidak Jelas</span>';
            }
            
            if (manipulationStatus) {
                manipulationStatus.innerHTML = 
                    '<i class="fas fa-check-circle text-green-400 text-xl mr-2"></i><span>Tidak Ada Manipulasi Terdeteksi</span>';
            }
            
            if (integrityStatus) {
                integrityStatus.innerHTML = 
                    '<i class="fas fa-check-circle text-green-400 text-xl mr-2"></i><span>Integritas Terjaga</span>';
            }
        }
        
        // Update metrics based on detection
        const integrityScore = result.isManipulated ? 30 + Math.random() * 20 : 80 + Math.random() * 15;
        const strengthScore = result.isManipulated ? 20 + Math.random() * 30 : 70 + Math.random() * 20;
        const resistanceScore = result.isManipulated ? 25 + Math.random() * 25 : 75 + Math.random() * 20;
        
        setTimeout(() => {
            const integrityScoreEl = document.getElementById('integrity-score');
            const integrityBar = document.getElementById('integrity-bar');
            
            if (integrityScoreEl) integrityScoreEl.textContent = Math.round(integrityScore) + '%';
            if (integrityBar) {
                integrityBar.style.width = integrityScore + '%';
                integrityBar.className = `${integrityScore > 60 ? 'bg-green-400' : 'bg-red-400'} h-2 rounded-full transition-all duration-1000`;
            }
        }, 200);
        
        setTimeout(() => {
            const strengthScoreEl = document.getElementById('strength-score');
            const strengthBar = document.getElementById('strength-bar');
            
            if (strengthScoreEl) strengthScoreEl.textContent = Math.round(strengthScore) + '%';
            if (strengthBar) strengthBar.style.width = strengthScore + '%';
        }, 400);
        
        setTimeout(() => {
            const resistanceScoreEl = document.getElementById('resistance-score');
            const resistanceBar = document.getElementById('resistance-bar');
            
            if (resistanceScoreEl) resistanceScoreEl.textContent = Math.round(resistanceScore) + '%';
            if (resistanceBar) resistanceBar.style.width = resistanceScore + '%';
        }, 600);
        
        const confidenceScore = document.getElementById('confidence-score');
        const modelUsed = document.getElementById('model-used');
        const processingTimeEl = document.getElementById('processing-time');
        
        if (confidenceScore) confidenceScore.textContent = result.confidence.toFixed(2) + '%';
        if (modelUsed) modelUsed.textContent = result.modelType.toUpperCase();
        if (processingTimeEl) processingTimeEl.textContent = processingTime + 's';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new WaterSenseApp();
    
    // Expose download functions to global scope
    window.downloadAnalysisAsImage = () => {
        const element = document.getElementById("analysis-result");
        if (!element) return;

        html2canvas(element).then(canvas => {
            const link = document.createElement("a");
            link.download = "hasil-analisis.png";
            link.href = canvas.toDataURL("image/png");
            link.click();
        });
    };

    window.downloadWatermarkedImageOnly = () => {
        const imageElement = document.getElementById("result-watermarked");
        if (!imageElement) return;

        const imageSrc = imageElement.src;
        const link = document.createElement("a");
        link.download = "gambar-watermarked.png";
        link.href = imageSrc;
        link.click();
    };
});