import tensorflow as tf


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_size = len(x_train)
    test_size = len(x_test)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((x_train, y_train))

    x_test = tf.data.Dataset.from_tensor_slices(x_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    test_dataset = tf.data.Dataset.zip((x_test, y_test))

    return train_dataset, test_dataset, train_size, test_size


def flatten_function(image, label):
    flattened_image = tf.reshape(image, [-1])
    norm_flat_image = flattened_image / 255

    return tf.cast(norm_flat_image, tf.float32), \
           tf.cast(label, tf.int32)
