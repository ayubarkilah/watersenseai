from flask import Flask, request, jsonify
import numpy as np
from scipy.linalg import svd

app = Flask(__name__)

@app.route('/api/watermark', methods=['POST'])
def apply_watermark():
    image = request.files['image'].read()
    watermark = request.form['watermark']
    strength = float(request.form['strength'])
    
    # Proses SVD di Python (Scipy)
    U, S, Vh = svd(image)
    watermarked_S = S + (watermark * strength)
    watermarked_image = U @ np.diag(watermarked_S) @ Vh
    
    return jsonify({"status": "success", "image": watermarked_image.tolist()})