import os
import gc
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
    # Load model efficiently
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model = None

# --- CONSTANTS (Must match training) ---
# Ensure this list is in the EXACT order of your training class_indices
CLASSES = ["COPD", "Asthma", "Pneumonia", "URTI", "Healthy", "Bronchiectasis", "LRTI", "Bronchiolitis"]
NUM_FEATURES = 52
MAX_FRAMES = 100

def extract_features(filepath):
    try:
        # MEMORY OPTIMIZATION: Limit duration to 10 seconds to save RAM on Free Tier
        y, sr = librosa.load(filepath, sr=22050, duration=10) 
        
        # Feature extraction
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NUM_FEATURES)
        log_spectrogram = librosa.power_to_db(spectrogram).T
        
        # Pad/Truncate
        if log_spectrogram.shape[0] > MAX_FRAMES:
            log_spectrogram = log_spectrogram[:MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - log_spectrogram.shape[0]
            log_spectrogram = np.pad(log_spectrogram, ((0, pad_width), (0, 0)), mode='constant')
            
        return log_spectrogram
    except Exception as e:
        print(f"Error parsing audio: {e}")
        return None
    finally:
        # Force garbage collection to free memory immediately
        gc.collect()

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if not model:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        try:
            # 1. Preprocess
            features = extract_features(filepath)
            if features is None:
                return jsonify({'error': 'Could not process audio file'}), 500

            # 2. Reshape for model (1, 100, 52)
            features = np.expand_dims(features, axis=0)

            # 3. Predict
            prediction = model.predict(features)
            
            # --- DEBUG LOGGING (Check Render Logs!) ---
            print("\n--- MODEL DIAGNOSIS ---")
            scores = {}
            for i, score in enumerate(prediction[0]):
                percent = score * 100
                scores[CLASSES[i]] = f"{percent:.2f}%"
                print(f"{CLASSES[i]}: {percent:.2f}%")
            print("-----------------------")
            
            # 4. Get Result
            predicted_index = np.argmax(prediction)
            predicted_class = CLASSES[predicted_index]
            confidence = float(prediction[0][predicted_index])

            return jsonify({
                'class': predicted_class,
                'confidence': f"{confidence * 100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'error': f"Server Error: {str(e)}"}), 500
        finally:
            # Clean up file and memory
            if os.path.exists(filepath):
                os.remove(filepath)
            gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)