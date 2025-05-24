// Muat model TensorFlow.js (misal: EfficientNet untuk deteksi deepfake)
let model;
async function loadModel() {
  model = await tf.loadLayersModel('static/js/tf-model/model.json');
  console.log("Model loaded!");
}

// Deteksi apakah gambar asli atau hasil AI
async function detectAIManipulation(imageElement) {
  const tensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims();
  
  const prediction = await model.predict(tensor).data();
  return prediction[0] > 0.5 ? "MANIPULATED" : "ORIGINAL";
}

// Event listener untuk upload gambar
document.getElementById('detection-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const image = await loadImage(file);
  const result = await detectAIManipulation(image);
  alert(`Hasil Deteksi: ${result}`);
});