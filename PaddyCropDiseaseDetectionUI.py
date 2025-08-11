import os
from flask import Flask, request, render_template
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained SVM model
try:
    with open('models/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
except FileNotFoundError:
    print("Error: svm_model.pkl not found. Please ensure the file is in the 'models' directory.")
    svm_model = None

# Load the VGG16 feature extractor directly
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Define the image size for VGG16
IMG_SIZE = (224, 224)

# Define the class names based on your LabelEncoder
# This dictionary is a corrected version based on your training script output.
class_to_disease = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy",
    3: "Leaf Smut",
    4: "Narrow Brown Spot",
    5: "Rice Blast"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="Error: No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="Error: No selected file")

    if file and svm_model:
        # Preprocess the uploaded image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize(IMG_SIZE)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Normalize

        # Extract features using VGG16
        features = feature_extractor.predict(image_array)
        features_flat = features.reshape(features.shape[0], -1)

        # Make a prediction with the SVM model
        prediction_index = int(svm_model.predict(features_flat)[0])
        
        # Use the dictionary to get the correct disease name, skipping the .ipynb_checkpoints
        try:
            # if prediction_index == 0:
            #     prediction_label = "System file detected, please upload a paddy leaf image."
            # else:
             prediction_label = class_to_disease[prediction_index]
        except KeyError:
            # Handle cases where the model predicts an unexpected index
            prediction_label = "Prediction failed: Unknown disease."

        return render_template('index.html', prediction_text=f'Predicted disease: {prediction_label}')
    
    return render_template('index.html', prediction_text="Error: Model not loaded.")

if __name__ == '__main__':
    app.run(debug=True)