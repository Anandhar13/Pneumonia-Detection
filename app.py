from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os


class PneumoniaDetector:
    def __init__(self, model_path):

        # Initialize the PneumoniaDetector Class.


        self.app = Flask(__name__)  # Create a Flask app instance
        self.model = load_model(model_path)  # Load the pre-trained model
        self.setup_routes()  # Set up the Flask routes

    def setup_routes(self):
        # Define the routes for the Flask application.

        @self.app.route('/', methods=['GET'])
        def home():

            # Render the home page and  HTML template for the home page.

            return render_template('index.html')

        @self.app.route('/', methods=['POST', 'GET'])
        def predict():

            # Handle image upload and make predictions.

            static_dir = './static/'
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)  # Create the directory if it doesn't exist

            imagefile = request.files["imagefile"]  # Get the uploaded image file
            image_path = os.path.join(static_dir, imagefile.filename)  # Define the save path for the image
            imagefile.save(image_path)  # Save the uploaded image

            # Preprocess the image
            img = load_img(image_path, target_size=(256, 256))  # Load and resize the image
            x = img_to_array(img)  # Convert the image to an array
            x = x / 255  # Normalize the image
            x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input

            # Predict the class using the model
            classes = self.model.predict(x)
            result1 = classes[0][0]  # Get the first class probability
            result2 = 'Negative'  # Default classification
            if result1 >= 0.5:  # If probability >= 0.5, classify as Positive
                result2 = 'Positive'
            classification = '%s (%.2f%%)' % (result2, result1 * 100)  # Format the output

            # Render the result in the template
            return render_template('index.html', prediction=classification, imagePath=image_path)

    def run(self, port=5000, debug=True):
        """
        Run the Flask application.

        Args:
            port (int): Port number to run the server.
            debug (bool): Whether to run in debug mode.
        """
        self.app.run(port=port, debug=debug)


# Instantiate the PneumoniaDetector class and start the application
if __name__ == '__main__':
    model_path = 'models/pnemonia_model.h5'  # Path to the trained model
    detector = PneumoniaDetector(model_path)  # Initialize the detector
    detector.run()  # Run the app
