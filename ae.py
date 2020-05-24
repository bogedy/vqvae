import tensorflow as tf
from tensorflow.keras.backend import batch_flatten

import os

optimizer = tf.optimizers.Adam(1e-4)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

class ae(tf.keras.Model):
    def __init__(self):
        super(ae, self).__init__()

        self.encoder = tf.keras.Sequential(
                [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4),
                tf.keras.layers.Dense(2),
                ]
                )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.Dense(4),
            tf.keras.layers.Dense(28*28),
            tf.keras.layers.Reshape(target_shape=(28, 28)),
            ]
            )

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def saver(self, tag):
        directory = './saved/{0}'.format(tag)
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.encoder.save(directory+'/inf', save_format='h5')
        self.decoder.save(directory+'/gen', save_format='h5')

@tf.function
def mse(input, output):
    #flatten the tensors, maintaining batch dim
    return tf.losses.MSE(batch_flatten(input), batch_flatten(output))

@tf.function
def train_step(input, model):
    with tf.GradientTape() as tape:
        z = model.encode(input)
        output = model.decode(z)
        print(input.shape, output.shape)
        loss = mse(input, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def train(model, inputs, n):
    for i in range(n):
        if i%100==0:
            print("Step {0}".format(i))
        train_step(inputs[2*i:2*(i+1)], model)

model = ae()

train(model, x_train, 100)
