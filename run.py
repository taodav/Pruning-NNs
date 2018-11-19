import tensorflow as tf
from argparse import ArgumentParser

from train import SparseTrainer
from model import SparseNN
from data import load_mnist, flatten_function

def get_args():
    parser = ArgumentParser(description='Pruning NNs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('--ptype', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./checkpoints/')

    description_excluded = ['save_path', 'save_every', 'log_every']
    args, _ = parser.parse_known_args()

    args.description = make_description(args, description_excluded)

    return args


def make_description(args, exclude):
    """
    Make a description for our checkpoints.
    :param args: arguments from argparse
    :param exclude: arguments to exclude from argparse
    :return: description of experiment run
    """
    all_args = vars(args).items()
    included_args = [k + "_" + str(v) for k, v in all_args if k not in exclude]
    return '-'.join(included_args)

if __name__ == "__main__":
    tf.enable_eager_execution()
    args = get_args()

    # load model
    model = SparseNN(ptype=args.ptype)

    # load dataset
    train_unparsed_dataset, test_unparsed_dataset, train_size, test_size = load_mnist()
    train_dataset = train_unparsed_dataset.map(flatten_function)

    # shuffle and batch dataset
    batched_train = train_dataset.shuffle(buffer_size=train_size).batch(args.batch_size)

    # initialize trainer
    trainer = SparseTrainer(model, batched_train, args)

    trainer.train()

