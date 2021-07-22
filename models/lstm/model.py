
import tensorflow as tf
from keras.utils.vis_utils import plot_model

def build(config, lstm_layers = [ 16, 32], dropout_layer = 0.5,
                     dense_layer = 32 ) -> tf.keras.Model :
    """Build LSTM Net Neural

    Args:
        config ([type]): [description]
        lstm_layers (list, optional): [description]. Defaults to [ 16, 32].
        dropout_layer (float, optional): [description]. Defaults to 0.5.
        dense_layer (int, optional): [description]. Defaults to 32.

    Returns:
        tf.keras.Model: [description]
    """
    input = tf.keras.Input(shape=(config.max_length))
    x = tf.keras.layers.Embedding(config.vocab_size, config.embedding_dim, input_length=config.max_length)(input)

    x = tf.keras.layers.LSTM(lstm_layers[0], return_sequences=True)(x)
    x = tf.keras.layers.LSTM(lstm_layers[1])(x)
    
    x = tf.keras.layers.Dropout(dropout_layer)(x)
    x = tf.keras.layers.Dense(dense_layer, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(input, output)

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model