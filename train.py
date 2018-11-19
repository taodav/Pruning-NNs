import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tqdm import tqdm

from helpers import get_prefix

class SparseTrainer:
    def __init__(self, model, train_dataset, args):
        """
        Sparse NN trainer
        :param model: Model to run
        :param train_dataset: process and batched training set
        :param args: arguments from argparse
        """
        self.model = model
        self.train_dataset = train_dataset
        self.args = args

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)

        self.summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis = 1000)

    def train(self):
        """
        Training loop.
        """
        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            for ep in range(self.args.epochs):
                # keep track of metrics we need
                loss_avg = tfe.metrics.Mean()
                accuracy = tfe.metrics.Accuracy()
                max_acc = 0

                tqdm.write("epoch %d" % ep)
                pbar = tqdm(enumerate(tfe.Iterator(self.train_dataset), 1))
                for batch, data in pbar:
                    images, labels = data

                    # take a training step
                    loss, dists = self.train_step(data)

                    # calculate metrics
                    loss_avg(loss)
                    accuracy(tf.argmax(dists, axis=1, output_type=tf.int32), labels)

                    if batch % self.args.log_every == 0:
                        print_loss = loss_avg.result()
                        print_acc = accuracy.result()

                        tf.contrib.summary.scalar('loss', print_loss)
                        tf.contrib.summary.scalar('accuracy', print_acc)

                        pbar.set_description("ep: %d, batch: %d, loss: %.4f, acc: %.4f" %\
                                             (ep, batch, print_loss, print_acc))

                    if (batch % self.args.save_every == 0) and accuracy.result() > max_acc:
                        # we save only the best (training) accuracy model
                        max_acc = accuracy.result()
                        prefix, _ = get_prefix(self.args)
                        checkpoint.save(file_prefix=prefix)


    def grad(self, inputs, targets):
        """
        Calculate gradients based on inputs and targets
        :param inputs: what to feed into model
        :param targets: target labels
        :return: gradients to apply, calculated loss and prediction
        """
        with tf.GradientTape() as tape:
            logits, dists = self.model(inputs)
            loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)
        return tape.gradient(loss, self.model.variables), loss.numpy(), dists

    def train_step(self, data):
        """
        Training step. Calculate and apply gradients over model variables
        :param data: data point to train on
        :return: calculated loss and distribution over classes from our model.
        """
        grads, loss, dists = self.grad(*data)
        self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                       global_step=tf.train.get_or_create_global_step())
        return loss, dists