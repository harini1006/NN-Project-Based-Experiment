# Project Based Experiments:
# Developed By: HARINI V
# Register Number: 212222230044
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```PY
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('img6n.png')
type(img)
img = image.load_img('img6n.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img1 = image.load_img('img9.png')
img_tensor1 = tf.convert_to_tensor(np.asarray(img1))
img_28_gray1 = tf.image.resize(img_tensor1,(28,28))
img_28_gray1 = tf.image.rgb_to_grayscale(img_28_gray1)
img_28_gray_inverted1 = 255.0-img_28_gray1
img_28_gray_inverted_scaled1 = img_28_gray_inverted1.numpy()/255.0

x_single_prediction1 = np.argmax(
    model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction1)
```
## Output:
# Training Loss, Validation Loss Vs Iteration Plot:
![327659992-b42a8f34-9887-40a4-9e0c-c8617a4e6c20](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/537c825d-7c4f-4565-bdc3-3afb1bfa182a)
![327660288-040f07e6-f06a-4930-9238-6492790fbf18](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/6093ea68-4fdc-4d01-b877-d05fef0bdfd4)
![327660450-e5ed9bc9-efc4-447d-b1f0-3df6f8e4e3d8](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/169d4820-7704-48a3-bf86-ac52bcaa8ea4)


# Classification Report:
![327660685-17c5ff7b-2bf2-4c50-9c97-93c2030a1c54](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/e52f438c-c5c0-4924-bcc3-5da5222d82b4)

# Confusion Matrix:
![327660908-d5361be7-0992-4ef1-b697-160100c0237e](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/a26d290d-3eb1-450e-a0ff-14b441d5f0ea)

# New Sample Data Prediction:

![327661315-5124e21e-019b-46da-aba0-6b02ca0ade63](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/12bae146-384b-4a28-949c-fc75266ac23f)

![327661440-3eb14d16-b0b5-4a3c-a266-bcd622e9240f](https://github.com/harini1006/NN-Project-Based-Experiment/assets/113497405/3a4e8be3-2524-4728-9d9e-f39a0105f061)


# Result:
Thus, a Multilayer Perceptron (MLP) to classify handwritten digits in python is build.







