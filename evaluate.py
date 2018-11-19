import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tqdm import tqdm

from run import get_args, flatten_function
from model import SparseNN
from data import load_mnist
from helpers import get_prefix

def evaluate_model(model, dataset):
    pbar = tqdm(enumerate(tfe.Iterator(dataset), 1))
    accuracy = tfe.metrics.Accuracy()

    for batch, data in pbar:
        images, labels = data
        logits, dists = model(images)

        accuracy(tf.argmax(dists, axis=1, output_type=tf.int32), labels)
    return accuracy.result()

def test_k(args, k_values):
    train_unparsed_dataset, test_unparsed_dataset, train_size, test_size = load_mnist()
    test_dataset = test_unparsed_dataset.map(flatten_function)
    batched_test = test_dataset.shuffle(buffer_size=test_size).batch(args.batch_size)

    checkpoint_prefix, checkpoint_dir = get_prefix(args)
    model = SparseNN(ptype=args.ptype)
    checkpoint = tf.train.Checkpoint(model=model)


    weights_accuracies = []
    for k in k_values:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        model.set_params(k / 100, "weights")
        weights_accuracies.append(evaluate_model(model, batched_test))

    nodes_accuracies = []
    for k in k_values:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        model.set_params(k / 100, "nodes")
        nodes_accuracies.append(evaluate_model(model, batched_test))

    return weights_accuracies, nodes_accuracies

if __name__ == "__main__":
    tf.enable_eager_execution()
    args = get_args()


    k_values = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]
    weights_accuracies, nodes_accuracies = test_k(args, k_values)

    print(weights_accuracies)
