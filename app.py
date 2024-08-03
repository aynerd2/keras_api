from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to the saved model
#Fix this path reference
# MODEL_PATH = r'C:\Users\Admin\Documents\nwexp\bck\model\model.h5'

MODEL_PATH = r'./model/model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return "Hello! This is the home page of the API."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file:
        # Save the file temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Load the image and prepare it for prediction
        img = load_img(filepath, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Remove the temporary file
        os.remove(filepath)

        return jsonify({'class': int(predicted_class), 'confidence': confidence})

    return jsonify({'error': 'File not readable'}), 400
 

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0',port=5000, debug=True)
