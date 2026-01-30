import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = "respiratory_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'respiratory_model.h5' is in the same directory.")

# Constants (Must match your training config)
CLASSES = ["COPD", "Asthma", "Pneumonia", "URTI", "Healthy", "Bronchiectasis", "LRTI", "Bronchiolitis"]
NUM_FEATURES = 52
MAX_FRAMES = 100

# --- PREPROCESSING FUNCTIONS ---
# These are copied from your script to ensure the input data matches the training data
def extract_log_spectrogram(filepath):
    try:
        signal, sr = librosa.load(filepath, sr=22050)
        spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=NUM_FEATURES)
        log_spectrogram = librosa.power_to_db(spectrogram).T
        
        # Padding/Truncating
        if log_spectrogram.shape[0] > MAX_FRAMES:
            log_spectrogram = log_spectrogram[:MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - log_spectrogram.shape[0]
            log_spectrogram = np.pad(log_spectrogram, ((0, pad_width), (0, 0)), mode='constant')
            
        return log_spectrogram
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess
        features = extract_log_spectrogram(filepath)
        
        if features is None:
             return jsonify({'error': 'Error processing audio file'})

        # Prepare for model (Add batch dimension: (1, 100, 52))
        features = np.expand_dims(features, axis=0)

        # Predict
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_class = CLASSES[predicted_index]
        confidence = float(prediction[0][predicted_index])

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)