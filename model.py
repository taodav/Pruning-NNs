import tensorflow as tf

class SparseNN(tf.keras.Model):
    def __init__(self):
        super(SparseNN, self).__init__()
        self.sparse_dense1 = SparseDense(1000)
        self.sparse_dense2 = SparseDense(1000)
        self.sparse_dense3 = SparseDense(500)
        self.sparse_dense4 = SparseDense(200)
        self.dense = tf.keras.layers.Dense(10, use_bias=False)

    def call(self, inputs):
        out1 = tf.nn.relu(self.sparse_dense1(inputs))
        out2 = tf.nn.relu(self.sparse_dense2(out1))
        out3 = tf.nn.relu(self.sparse_dense3(out2))
        out4 = tf.nn.relu(self.sparse_dense4(out3))
        logits = self.dense(out4)
        dist = tf.nn.softmax(logits)

        return logits, dist


class SparseDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(SparseDense, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.weight = self.add_variable("weight",
                                        shape=[input_shape[-1].value, self.num_outputs])


    def call(self, input):
        return tf.matmul(input, self.weight)

