Loading and Preparing Data:

The code loads brain tumor images from specified folders for training and testing.
It resizes these images to a fixed size (224x224 pixels).
The images are then stored in arrays (X_train) along with their corresponding labels (Y_train).
The data is shuffled and split into training and testing sets.
Processing Labels:

The labels (like 'glioma_tumor', 'meningioma_tumor', etc.) are converted from strings to numerical format (e.g., 0, 1, 2, 3).
These numerical labels are then one-hot encoded, which is a common format required for multi-class classification tasks.
Building a Convolutional Neural Network (CNN):

A sequential model is created using Keras.
Several convolutional layers (Conv2D) are added to extract features from the images.
The model includes pooling layers (MaxPooling2D) to reduce the dimensionality of the feature maps.
Batch normalization layers are added to stabilize and speed up training.
DepthwiseConv2D layers are included to make the model more efficient by reducing the number of parameters.
The model ends with a global average pooling layer, dense (fully connected) layers, and a final output layer with softmax activation to classify the images into one of the four tumor types.
Compiling the Model:

The model is compiled using the categorical cross-entropy loss function, which is standard for multi-class classification.
The Adam optimizer is used to adjust the model's weights based on the loss.
Accuracy is used as the metric to evaluate the model's performance during training.
Training the Model:

The model is trained using the training data, and its performance is validated on a small portion of the training data.
A callback (ReduceLROnPlateau) is used to automatically reduce the learning rate if the model's validation loss stops improving, helping to fine-tune the training process.
Evaluating the Model:

After training, the model's performance is evaluated on the test data, and the test loss and accuracy are printed.
Saving the Model:

The model's architecture is saved as a JSON file, and the trained weights are saved in an H5 file for future use or deployment.
In essence, this code is about building, training, and saving a deep learning model that can classify different types of brain tumors from medical images
