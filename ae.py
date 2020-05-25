import tensorflow as tf
from tensorflow.keras.backend import batch_flatten

import os
from tqdm import tqdm

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
        loss = mse(input, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def train(model, inputs, batch_size):
    for i in tqdm(range(int(inputs.shape[0]/batch_size))):
        train_step(inputs[batch_size*i:batch_size*(i+1)], model)

if __name__ == "__main__":
    
    model = ae()

    train(model, x_train, 3)


