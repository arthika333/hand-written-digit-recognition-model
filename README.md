# Hand-written-digit-recognition-model
This is a Convolutional Neural Network model which recognizes hand written digits. It is trained and tested on the tensorflow MNIST dataset and achieved **97.8%** test accuracy.

## Purpose and Goals
The purpose of this project is to build and train a Convolutional Neural Network (CNN) capable of recognizing handwritten digits using the MNIST dataset. It also extends beyond the tensorflow dataset by testing predictions on custom handwritten digits, utilizing preprocessing to impact the accuracy.

The goal was to achieve high test accuracy of over 95%, demonstrate real-world applicability on non-MNIST inputs, and document the process in a clear, reproducible way. This model serves as an initial project in deep learning.

## Model Performance
-**Training Accuracy:** 99%\
-**Testing Accuracy:** 97.8%\
-**Prediction Accuracy:** 15/16 accuracte predictions on custom digits

## Required
- Python 3.12.11
- TensorFlow 2.8
- NumPy
- Matplotlib

## Image Preprocessing
All images of the MNIST Tensorflow data set were normalized to make sure all of their values existed between zero and 1. The custom images were also normalized in the same manner with the added step of preprocessing the images to invert the colors so that digits appeared white on a black background (to match the MNIST format).

## Model Structure, Compilation and Fitting

The model was built using TensorFlow’s Sequential API.
- Input Layer: A Flatten layer to convert the 28×28 pixel images into 1D vectors.
- Hidden Layers: Two fully connected dense layers with 128 neurons each, using ReLU activation.
- Output Layer: A Dense layer with 10 neurons (one per class), using softmax activation.
<img width="949" height="418" alt="image" src="https://github.com/user-attachments/assets/0001a521-9467-40ac-90cf-c4b07df10b7b" />

The model was compiled with:
- Loss: Sparse Categorical Crossentropy (since labels were integers, not one-hot encoded)
- Optimizer: Adam (using default learning rate)
- Metric: Accuracy

The model was trained for 20 epochs on the normalized dataset, with shuffling enabled. No additional callbacks were used in order to keep the model simple. 

## Model Evaluation
After 20 epochs of training, the model achieved:
- Training Accuracy: 99.7%
- Training Loss: ~0.9%
Evaluating on the test dataset, the model achieved:
- Test Accuracy: 97.8%
- Test Loss: 0.16

These results remained consistent when the model was saved, reloaded, and re-evaluated.

## Model Predictions
### on MNIST data
<img width="1347" height="499" alt="image" src="https://github.com/user-attachments/assets/093b9b33-2654-4618-8407-26e450e8f943" />
Random predictions on test samples consistently matched the true labels. Misclassifications were rare.

### on custom data
<img width="1320" height="478" alt="image" src="https://github.com/user-attachments/assets/69e3a597-7819-464f-8357-80c56a266027" />
<img width="1330" height="485" alt="image" src="https://github.com/user-attachments/assets/5b21a077-2ddb-4e20-9608-d578f9c435a5" />
Custom 28×28 hand-drawn digit images were preprocessed (centering the digit and inverting colors) before prediction. Out of 16 samples, the model correctly predicted 15. The only misclassification was a digit "9" being predicted as a "3".

## Clone this repository
```
git clone https://github.com/yourusername/Hand-written-digit-recognition-model.git
cd Hand-written-digit-recognition-model
```
## Train and test the model
```
pip install -r requirements.txt
python src/training.py
python src/utils.py
python src/predicting.py
```

