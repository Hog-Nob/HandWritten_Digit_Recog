Handwritten Digit Recognition Web App

This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model predicts the digit from a provided image of a handwritten digit (0-9).

Key Features
- Upload an image of a handwritten digit (0-9).
- The web app uses a trained CNN model to predict the digit.
- Displays the predicted digit on the web interface.

Technologies Used
- TensorFlow/Keras: For training and using the CNN model.
- Streamlit: For building the interactive web interface.
- NumPy and PIL: For image processing.

Installation
- Clone the repository or download the project files.
- Create a virtual environment and activate it:
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
- Install the required dependencies:
pip install -r requirements.txt
- If opencv-python is missing, install it:
pip install opencv-python

Running the App
- Ensure all dependencies are installed.
- Run the app using Streamlit:
streamlit run app.py
- Open your browser and navigate to the URL provided by Streamlit to use the web app.
  
Model Details
- The model is a CNN trained on the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits.
- The model has been saved in HDF5 format (mnist_cnn_model.h5).

Contributing
- Feel free to fork the repository, submit pull requests, or suggest improvements.

License
- This project is licensed under the MIT License - see the LICENSE file for details.
