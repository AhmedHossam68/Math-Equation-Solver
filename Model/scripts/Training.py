import tensorflow as tf
import tf_keras as k
import numpy as np
import ModelArchitecture
import DataGenerator





# Constants
NB_EPOCHS = 50
BATCH_SIZE = 20
BEST_PERPLEXITY = float('inf')
EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = 512
H = 20
W = 50
num_feats = 64
D = num_feats * 8
V = 502  # Vocabulary size

# Define the model
class MyModel(k.Model):
    def __init__(self, num_feats, bn, input_dim, ENC_DIM, DEC_DIM, D, H, W):
        super(MyModel, self).__init__()
        self.cnn = ModelArchitecture.cnn_architecture
        self.enc_dec = ModelArchitecture.EncDecAttention
        self.num_feats = num_feats
        self.bn = bn
        self.input_dim = input_dim
        self.ENC_DIM = ENC_DIM
        self.DEC_DIM = DEC_DIM
        self.D = D
        self.H = H
        self.W = W

    def call(self, inputs):
        X = self.cnn(inputs, self.num_feats, self.bn)
        outputs = self.enc_dec(inputs, X, self.input_dim, self.ENC_DIM, self.DEC_DIM, self.D, self.H, self.W)
        return outputs

# Instantiate the model
model = MyModel(num_feats, True, EMB_DIM, ENC_DIM, DEC_DIM, D, H, W)

# Define the optimizer
optimizer = k.optimizers.Adam()

# Define the loss function
loss_fn = k.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define the metric
metric = k.metrics.SparseCategoricalAccuracy()

# Define the training step
@tf.function
def train_step(imgs, Y, mask):
    with tf.GradientTape() as tape:
        predictions = model(imgs)
        loss = loss_fn(Y, predictions, sample_weight=mask)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric.update_state(Y, predictions, sample_weight=mask)
    return loss

# Define the training loop
def train_model(data_loader, epochs):
    for epoch in range(epochs):
        for imgs, Y, mask in data_loader:
            loss = train_step(imgs, Y, mask)
        print(f'Epoch {epoch + 1}, Loss: {loss}, Accuracy: {metric.result()}')
        metric.reset_states()

# Train the model
train_model(DataGenerator.data_loader('training_list', BATCH_SIZE), NB_EPOCHS)