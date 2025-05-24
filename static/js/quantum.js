/**
 * Watermarking dengan pendekatan quantum-inspired (DCT + SVD)
 */
function quantumWatermark(imageData, watermark) {
    // 1. Transformasi DCT (Domain Frekuensi)
    const dctData = applyDCT(imageData); 
  
    // 2. Ambil komponen frekuensi rendah & tinggi
    const { lowFreq, highFreq } = splitFrequencyComponents(dctData);
  
    // 3. Sisipkan watermark di kedua domain
    const watermarkedLow = embedInLowFreq(lowFreq, watermark);
    const watermarkedHigh = embedInHighFreq(highFreq, invertWatermark(watermark));
  
    // 4. Gabungkan dan inverse DCT
    return applyInverseDCT(combineComponents(watermarkedLow, watermarkedHigh));
  }