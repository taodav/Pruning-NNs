import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tqdm import tqdm

from helpers import get_prefix

class SparseTrainer:
    def __init__(self, model, train_dataset, test_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)

        self.summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis = 1000)

    def train(self):
        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            for ep in range(self.args.epochs):
                loss_avg = tfe.metrics.Mean()
                accuracy = tfe.metrics.Accuracy()
                max_acc = 0

                tqdm.write("epoch %d" % ep)
                pbar = tqdm(enumerate(tfe.Iterator(self.train_dataset), 1))
                for batch, data in pbar:
                    images, labels = data
                    loss, dists = self.train_step(data)
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
                        max_acc = accuracy.result()
                        checkpoint.save(file_prefix=get_prefix(self.args))


    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            logits, dists = self.model(inputs)
            loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)
        return tape.gradient(loss, self.model.variables), loss.numpy(), dists

    def train_step(self, data):
        grads, loss, dists = self.grad(*data)
        self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                       global_step=tf.train.get_or_create_global_step())
        return loss, dists