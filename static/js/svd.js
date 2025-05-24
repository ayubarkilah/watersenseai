import * as numeric from 'https://cdn.jsdelivr.net/npm/numeric@1.2.6/numeric.min.js';

/**
 * Sisipkan watermark ke gambar menggunakan SVD
 * @param {ImageData} imageData - Data gambar dari canvas
 * @param {Uint8Array} watermark - Data watermark (binary)
 * @param {number} strength - Kekuatan watermark (0.1 - 0.5)
 */
function embedWatermarkSVD(imageData, watermark, strength = 0.1) {
  const pixels = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  
  // Pisahkan channel RGB
  const channels = { r: [], g: [], b: [] };
  for (let i = 0; i < pixels.length; i += 4) {
    channels.r.push(pixels[i]);
    channels.g.push(pixels[i + 1]);
    channels.b.push(pixels[i + 2]);
  }

  // Proses SVD per channel
  const watermarkedChannels = {};
  for (const [channel, data] of Object.entries(channels)) {
    const matrix = reshapeToMatrix(data, width, height);
    const { U, S, V } = numeric.svd(matrix);
    
    // Sisipkan watermark ke singular values (S)
    for (let i = 0; i < Math.min(watermark.length, S.length); i++) {
      S[i] += watermark[i] * strength;
    }
    
    // Rekonstruksi matriks
    watermarkedChannels[channel] = numeric.dot(
      numeric.dot(U, numeric.diag(S)),
      numeric.transpose(V)
    );
  }

  // Gabungkan kembali ke ImageData
  const watermarkedPixels = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    const row = Math.floor((i / 4) / width);
    const col = (i / 4) % width;
    watermarkedPixels[i] = watermarkedChannels.r[row][col];
    watermarkedPixels[i + 1] = watermarkedChannels.g[row][col];
    watermarkedPixels[i + 2] = watermarkedChannels.b[row][col];
    watermarkedPixels[i + 3] = pixels[i + 3]; // Alpha channel
  }

  return new ImageData(watermarkedPixels, width, height);
}

// Helper: Ubah array 1D ke matriks 2D
function reshapeToMatrix(data, width, height) {
  const matrix = [];
  for (let i = 0; i < height; i++) {
    matrix.push(data.slice(i * width, (i + 1) * width));
  }
  return matrix;
}