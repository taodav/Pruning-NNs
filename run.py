import tensorflow as tf
from argparse import ArgumentParser

from train import SparseTrainer


def get_args():
    parser = ArgumentParser(description='Pruning NNs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./checkpoints')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    tf.enable_eager_execution()
    args = get_args()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((x_train, y_train))

    x_test = tf.data.Dataset.from_tensor_slices(x_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    test_dataset = tf.data.Dataset.zip((x_test, y_test))
    