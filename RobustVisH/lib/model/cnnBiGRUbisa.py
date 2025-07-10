import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras import backend as K

'''x1-out1:model_haptic'''
def CNN_H(X1_train):
    x1 = keras.layers.Input(X1_train.shape[1:], name='Input_1')

    conv1 = keras.layers.Conv1D(128, 8, padding="same")(x1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    conv2 = keras.layers.Conv1D(256, 5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, 3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    pool1 = keras.layers.AveragePooling1D(pool_size=2)(conv3)
    rnn1 = keras.layers.Bidirectional(keras.layers.GRU(128), merge_mode='sum')(pool1)

    out1 = keras.layers.Reshape((128, 1))(rnn1)
    cnn1 = Model(x1, out1)
    return cnn1


'''x2-out2:model_kinesthetics'''
def CNN_K(X2_train):
    x2 = keras.layers.Input(X2_train.shape[1:], name='Input_2')

    conv4 = keras.layers.Conv1D(128, 8, padding="same")(x2)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)

    conv5 = keras.layers.Conv1D(256, 5, padding="same")(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation('relu')(conv5)

    conv6 = keras.layers.Conv1D(128, 3, padding="same")(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Activation('relu')(conv6)

    full2 = keras.layers.GlobalAveragePooling1D()(conv6)

    out2 = keras.layers.Reshape((128, 1))(full2)
    cnn2 = Model(x2, out2)
    return cnn2


'''x3-out3:model_visual'''
def CNN_V(X3_train):
    x3 = keras.layers.Input(X3_train.shape[1:], name='Input_3')

    conv7 = keras.layers.Conv1D(128, 8, padding="same")(x3)
    conv7 = keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Activation('relu')(conv7)

    conv8 = keras.layers.Conv1D(256, 5, padding="same")(conv7)
    conv8 = keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Activation('relu')(conv8)

    conv9 = keras.layers.Conv1D(128, 3, padding="same")(conv8)
    conv9 = keras.layers.BatchNormalization()(conv9)
    conv9 = keras.layers.Activation('relu')(conv9)

    full3 = keras.layers.GlobalAveragePooling1D()(conv9)
    
    out3 = keras.layers.Reshape((128, 1))(full3)
    cnn3 = Model(x3, out3)
    return cnn3

def bidirectional_transformer_encoder_layer(inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0.05, kernel_size: int = 1):
    """Bidirectional Encoder: Attention and Normalization and Feed-Forward."""
    # 1. MultiHeadAttention (Forward):
    x_forward = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x_forward = keras.layers.Dropout(dropout)(x_forward)

    # 2. MultiHeadAttention (Backward):
    # Reverse the inputs along the sequence axis (axis=1)
    reversed_inputs = tf.reverse(inputs, axis=[1])
    x_backward = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(reversed_inputs, reversed_inputs)
    x_backward = keras.layers.Dropout(dropout)(x_backward)
    # Reverse the backward output back to the original sequence order
    x_backward = tf.reverse(x_backward, axis=[1])

    # 3. Combine Forward and Backward outputs:
    x = keras.layers.Add()([x_forward, x_backward])

    # 4. Add&Norm:
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # 5. Feed Forward:
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)

    # 6. Add&Norm:
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def BiTransformerEncoder(inputs, head_size: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float = 0.05, kernel_size: int = 1,):
    x = inputs
    for i in range(num_layers):
        x = bidirectional_transformer_encoder_layer(x, head_size, num_heads, ff_dim, dropout, kernel_size)
    return x


class WeightedFusionUnit(layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedFusionUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='weights1', shape=(len(input_shape), 1), initializer='uniform', trainable=True)
        print(self.w, 'weights1')

    def call(self, inputs, **kwargs):
        return keras.backend.dot(keras.backend.stack(inputs, axis=-1), self.w)[..., -1]



def build_T(X1_train, X2_train, X3_train, nb_classes):
    x1 = keras.layers.Input(X1_train.shape[1:], name='Input_1')
    x2 = keras.layers.Input(X2_train.shape[1:], name='Input_2')
    x3 = keras.layers.Input(X3_train.shape[1:], name='Input_3')
    in1 = x1
    in2 = x2
    in3 = x3

    for i in range(1, len(CNN_H(X1_train).layers)):
        x1 = CNN_H(X1_train).layers[i](x1)
    for j in range(1, len(CNN_K(X2_train).layers)):
        x2 = CNN_K(X2_train).layers[j](x2)
    for k in range(1, len(CNN_V(X3_train).layers)):
        x3 = CNN_V(X3_train).layers[k](x3)

    merger = WeightedFusionUnit()([x1, x2, x3])
    print(merger, 'merge')

    BiT = BiTransformerEncoder(merger, head_size=1, num_heads=3, ff_dim=128, num_layers=2)

    full = layers.Dropout(0.5)(BiT)
    full = layers.Dense(1, activation='relu')(full)
    full = layers.Flatten(input_shape=(128, 1))(full)

    output = keras.layers.Dense(nb_classes, activation='softmax')(full)

    model = Model(inputs=[in1, in2, in3], outputs=output)
    return model
