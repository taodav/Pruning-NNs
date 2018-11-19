import tensorflow as tf
import numpy as np

class SparseNN(tf.keras.Model):
    def __init__(self, k=0.0, ptype=None):
        super(SparseNN, self).__init__()
        self.sparse_dense1 = SparseDense(1000, k, ptype=ptype)
        self.sparse_dense2 = SparseDense(1000, k, ptype=ptype)
        self.sparse_dense3 = SparseDense(500, k, ptype=ptype)
        self.sparse_dense4 = SparseDense(200, k, ptype=ptype)
        self.dense = tf.keras.layers.Dense(10, use_bias=False)

    def set_params(self, k, ptype):
        for layer in [self.sparse_dense1, self.sparse_dense2, self.sparse_dense3, self.sparse_dense4]:
            layer.k = k
            layer.ptype = ptype

    def call(self, inputs):
        out1 = tf.nn.relu(self.sparse_dense1(inputs))
        out2 = tf.nn.relu(self.sparse_dense2(out1))
        out3 = tf.nn.relu(self.sparse_dense3(out2))
        out4 = tf.nn.relu(self.sparse_dense4(out3))
        logits = self.dense(out4)
        dist = tf.nn.softmax(logits)

        return logits, dist


class SparseDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs, k, ptype="weights"):
        super(SparseDense, self).__init__()
        self.num_outputs = num_outputs
        self.k = k
        self.ptype = ptype

    def build(self, input_shape):
        self.weight = self.add_variable("weight",
                                        shape=[input_shape[-1].value, self.num_outputs])

    def call(self, input):
        if self.ptype == "weights":
            # first we calculate our weight mask
            weights = self.weight.numpy()
            weights_l1_norm = abs(weights)
            k_idx = int(weights.size * self.k)
            threshold = np.partition(weights_l1_norm.flatten(), k_idx)[k_idx]
            weight_mask = tf.cast(weights_l1_norm > threshold, tf.float32)
            self.weight.assign(self.weight * weight_mask)
        elif self.ptype == "nodes":
            weights_l2_norm = tf.norm(self.weight, ord=2, axis=0).numpy()
            k_idx = int(weights_l2_norm.size * self.k)
            threshold = np.partition(weights_l2_norm, k_idx)[k_idx]
            flat_mask = weights_l2_norm > threshold
            weight_mask = tf.cast(tf.stack([flat_mask for i in range(self.weight.shape[0].value)]), tf.float32)
            self.weight.assign(self.weight * weight_mask)

        return tf.matmul(input, self.weight)

