// SVD Watermarking Implementation
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

// Export untuk digunakan di file lain
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SVDWatermarking;
}