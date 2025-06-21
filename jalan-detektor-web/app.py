from flask import Flask, render_template, request
import os, cv2, joblib, base64
import numpy as np
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ===== Fungsi Ekstraksi Fitur =====
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return features, gray

def adjust_brightness(image, brightness=20):
    return cv2.convertScaleAbs(image, beta=brightness)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# ===== Route Utama =====
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded = request.files['image']
        filename = uploaded.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        uploaded.save(filepath)

        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (256, 256))
        img_bright = adjust_brightness(img_resized, 20)
        features, gray = extract_glcm_features(img_bright)
        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        label = "berlubang" if pred == 1 else "biasa"

        return render_template('index.html',
                               original=image_to_base64(img_resized),
                               bright=image_to_base64(img_bright),
                               gray=image_to_base64(gray),
                               features=features,
                               label=label)
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)