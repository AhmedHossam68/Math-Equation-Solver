import tensorflow as tf
import tf_keras as k
import numpy as np

print(tf.__version__)
 

def cnn_architecture(X, num_feats, bn, train_mode=True):
    
    X = X-128
    X= X/128

    X = k.layers.Conv2D(num_feats, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    X = k.layers.ReLU()(X)
    X = k.layers.MaxPooling2D((2, 2), 2, data_format="channels_first")(X)

    X = k.layers.Conv2D(num_feats*2, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    X = k.layers.ReLU()(X)
    X = k.layers.MaxPooling2D((2, 2), 2, data_format="channels_first")(X)

    X = k.layers.Conv2D(num_feats*4, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    if bn:
        X = k.layers.BatchNormalization(trainable=train_mode)(X)
    X = k.layers.ReLU()(X)

    X = k.layers.Conv2D(num_feats*4, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    X = k.layers.ReLU()(X)
    X = k.layers.MaxPooling2D((1, 2), (1,2), data_format="channels_first")(X)

    X = k.layers.Conv2D(num_feats*8, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    if bn:
        X = k.layers.BatchNormalization(trainable=train_mode)(X)
    X = k.layers.ReLU()(X)
    X = k.layers.MaxPooling2D((2, 1), (2,1))(X)

    X = k.layers.Conv2D(num_feats*8, (3, 3), padding='same', use_bias=False, data_format="channels_first")(X)
    if bn:
        X = k.layers.BatchNormalization(trainable=train_mode)(X)
    X = k.layers.ReLU()(X)

    return X



class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, name, n_in, n_hid, L, D, ctx, forget_bias=1.0):
        super(AttentionCell, self).__init__(name=name)
        self._n_in = n_in
        self._n_hid = n_hid
        self._forget_bias = forget_bias
        self._ctx = ctx
        self._L = L
        self._D = D

        self.gates_layer = k.layers.Dense(4 * self._n_hid, activation='sigmoid')
        self.target_layer = k.layers.Dense(self._n_hid, use_bias=False)
        self.output_layer = k.layers.Dense(self._n_hid, activation='tanh', use_bias=False)

    def build(self):
        self.built = True

    def call(self, inputs, states):
        _input, output_tm1 = inputs
        h_tm1, c_tm1 = states

        gates = self.gates_layer(k.layers.concatenate([_input, output_tm1]))
        i_t, f_t, o_t, g_t = tf.split(gates, num_or_size_splits=4, axis=1)

        c_t = k.activations.sigmoid(f_t) * c_tm1 + k.activations.sigmoid(i_t) * k.activations.tanh(g_t)
        h_t = k.activations.sigmoid(o_t) * k.activations.tanh(c_t)

        target_t = tf.expand_dims(self.target_layer(h_t), 2)

        a_t = k.activations.softmax(tf.matmul(self._ctx, target_t)[:,:,0], axis=-1)
        a_t = tf.expand_dims(a_t, 1)
        z_t = tf.squeeze(tf.matmul(a_t, self._ctx), 1)


        output_t = self.output_layer(k.layers.concatenate([h_t, z_t]))

        return output_t, [h_t, c_t]

    def get_config(self):
        return {'name': self.name, 'n_in': self._n_in, 'n_hid': self._n_hid, 'L': self._L, 'D': self._D, 'ctx': self._ctx}

    @property
    def state_size(self):
        return [self._n_hid, self._n_hid]

    @property
    def output_size(self):
        return self._n_hid



def EncDecAttention(
    inputs,
    ctx,
    input_dim,
    ENC_DIM,
    DEC_DIM,
    D,
    H,
    W
    ):
    """
    Function that encodes the feature grid extracted from CNN using BiLSTM encoder
    and decodes target sequences using an attentional decoder mechanism

    PS: Feature grid can be of variable size (as long as size is within 'H' and 'W')

    :parameters:
        ctx - (N,C,H,W) format ; feature grid extracted from CNN
        input_dim - int ; Dimensionality of input sequences (Usually, Embedding Dimension)
        ENC_DIM - int; Dimensionality of BiLSTM Encoder
        DEC_DIM - int; Dimensionality of Attentional Decoder
        D - int; No. of channels in feature grid
        H - int; Maximum height of feature grid
        W - int; Maximum width of feature grid
    """
    V = tf.transpose(ctx, [0, 2, 3, 1])
    V_reshaped = tf.reshape(V, [-1, W, D])

    V_encoded = k.layers.Bidirectional(k.layers.LSTM(ENC_DIM, return_sequences=True))(V_reshaped)

    V_encoded = tf.reshape(V_encoded, [-1, H*W, 2*ENC_DIM])

    cell = AttentionCell(input_dim, DEC_DIM, H*W, 2*ENC_DIM, V_encoded)

    rnn_layer = k.layers.RNN(cell)
    outputs = rnn_layer(inputs)

    return outputs

def data_loader(set_type, batch_size):
    # Load numpy buckets into dictionary
    train_dict = np.load(set_type + '_buckets.npy').tolist()
    
    # Iterate over each bucket
    for img_size_group in train_dict.keys():
        # Create a list to hold image names and tokenized sequences
        train_list = train_dict[img_size_group]
        
        # Determine the number of files used for processing
        num_files = (len(train_list) // batch_size) * batch_size
        
        # Iterate over each batch
        for batch_idx in range(0, num_files, batch_size):
            # Create a sublist of the training list for the current batch
            train_sublist = train_list[batch_idx:batch_idx + batch_size]
            
            # Initialize lists for images and batch forms (labels)
            imgs = []
            batch_forms = []
            
            # Process each image and its corresponding tokenized sequence
            for img_name, tokenized_seq in train_sublist:
                # Load and preprocess the image
                img = np.asarray(Image.open('./images_processed/' + img_name).convert('YCbCr'))[:, :, 0][:, :, None]
                imgs.append(img)
                
                # Append the tokenized sequence to the batch forms list
                batch_forms.append(tokenized_seq)
            
            # Convert the list of images into a numpy array
            imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 3, 1, 2)
            
            # Calculate the lengths of all the formulas in the batch
            lens = [len(form) for form in batch_forms]
            
            # Initialize mask and Y arrays
            mask = np.zeros((batch_size, max(lens)), dtype=np.int32)
            Y = np.zeros((batch_size, max(lens)), dtype=np.int32)
            
            # Fill mask and Y arrays with data
            for i, form in enumerate(batch_forms):
                mask[i, :len(form)] = 1
                Y[i, :len(form)] = form
            
            # Yield the data to the model for processing
            yield imgs, Y, mask
