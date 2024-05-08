import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from cystMarker import highlight_cysts

app = Flask(__name__)

# Configure the UPLOAD_FOLDER
UPLOAD_FOLDER = 'D:/TY-SEM 6/Project/OvarianCystDetectionCNN/UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the UPLOAD_FOLDER directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model('ModelOvarianCystClassification.h5')

# Function to get the class name based on class number
def get_className(classNo):
    if classNo == 0:
        return "No Cyst present"
    elif classNo == 1:
        return "Cyst present"

# Function to preprocess and predict the image
def get_Result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    inputImg = np.expand_dims(image, axis=0)
    # Predict probabilities for each class
    predictions = model.predict(inputImg)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    return get_className(predicted_class)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get classification result
        result = get_Result(filepath)
        
        # Initialize highlighted image path
        highlighted_image_path = filepath
        
        # Check if cyst is present
        if result == "Cyst present":
            # Highlight cysts
            highlighted_image = highlight_cysts(filepath)
            
            # Save or serve the highlighted image as needed
            # For now, let's just save it back to the same file path
            highlighted_image_path = filepath.replace('.jpg', '_highlighted.jpg')
            cv2.imwrite(highlighted_image_path, highlighted_image)
        
        # Return classification result and highlighted image path
        return jsonify({'result': result, 'highlighted_image_path': highlighted_image_path}), 200


if __name__ == '__main__':
    app.run(debug=True)
