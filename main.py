import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, SparseCategoricalCrossentropy


fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
hidden_layers = [0, 1, 2]
widths = [64, 128, 256]
loss_functions = [MeanSquaredError(), MeanAbsoluteError(), SparseCategoricalCrossentropy()]


def build_model(hidden_layers, width, loss_function):
    model = tf.keras.models.Sequential()
    model.add(Dense(784, activation='relu'))
    for _ in range(hidden_layers):
        model.add(Dense(width, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=SGD(), loss=loss_function, metrics=['accuracy'])
    return model

def prepare_labels(loss_function, y_train, y_test):
    if isinstance(loss_function, (MeanSquaredError, MeanAbsoluteError)):
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    elif isinstance(loss_function, SparseCategoricalCrossentropy):
        # No need to modify labels for SparseCategoricalCrossentropy
        pass
    return y_train, y_test

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test), verbose=0)
    return history.history['loss'], history.history['accuracy'], history.history['val_accuracy']

for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_layer in hidden_layers:
            for width in widths:
                for loss_function in loss_functions:

                    # Build model

                    model = build_model(hidden_layer, width, loss_function)
                    # Prepare labels
                    y_train_prepared, y_test_prepared = prepare_labels(loss_function, y_train, y_test)
                    # Train and evaluate model
                    loss, train_accuracy, val_accuracy = train_and_evaluate_model(model, x_train, y_train_prepared, x_test, y_test_prepared, epochs=10, batch_size=batch_size)
                    # Plot loss and accuracy

                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(loss)
                    plt.title('Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.subplot(1, 2, 2)
                    plt.plot(train_accuracy, label='Training Accuracy')
                    plt.title('Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    plt.suptitle(f'LR={lr}, Batch={batch_size}, Layers={hidden_layer}, Width={width}, Loss={loss_function}')
                    plt.show()
