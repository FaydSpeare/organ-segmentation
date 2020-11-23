import tensorflow as tf

from tf_utils import residual_dense_block

def add_layer(model, layer):
    if not layer._layers:
        return model._layers.append(layer)
    for sub_layer in layer._layers:
        add_layer(model, sub_layer)


def full_summary(model, input_shape):

    copied_layers = model._layers.copy()
    model._layers.clear()
    for layer in copied_layers:
        add_layer(model, layer)

    model(tf.keras.Input(shape=input_shape))
    super(type(model), model).summary(line_length=200)
    model._layers = copied_layers


if __name__ == '__main__':
    block = residual_dense_block.ResDense(2, 32, 3, "he_uniform", batch_norm=True)
    full_summary(block, (8, 8, 8, 2))


#summary_model = tf.keras.Model(inputs=[x], outputs=model.call(x, training=False))
#tf.keras.utils.plot_model(model, to_file='./file.png', show_shapes=True, expand_nested=True)
#model.summary(line_length=200)

