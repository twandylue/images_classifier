import tensorflow as tf
import matplotlib.pyplot as plt


def train_model():
    """
    Train a simple CNN model on the MNIST dataset.
    """
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize the images to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(
        tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation="relu", input_shape=(28, 28, 1)
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))  # Add dropout to the fully connected layer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Compile and train the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )
    return model, history


def plot_result(model, history):
    """
    Plot the training and validation accuracy and loss
    """
    print(f"The learning rate is {model.optimizer.learning_rate.numpy():.3e}")
    print(f"The loss function is {model.loss.name}")
    print(f"The number of epochs is {len(history.history['accuracy'])}")
    print(f"The optimizer is {model.optimizer.get_config()['name']}")


if __name__ == "__main__":
    model, history = train_model()
    plot_result(model, history)

    # Save the model
    model.save("mnist_model.h5")
