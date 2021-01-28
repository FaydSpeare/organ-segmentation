import tensorflow as tf


class Losses:

    @staticmethod
    def get_loss_fn(params):
        name = params['loss_fn']
        if name == 'DICEL':
            return dice_loss
        elif name == 'CCE':
            return categorical_crossentropy
        elif name == 'SCCE':
            return sparse_categorical_crossentropy
        else:
            raise NotImplementedError(f"No loss function exists with name: '{name}'")



def categorical_crossentropy(y_true, y_pred, from_logits=False):
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits))


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits))


def dice_loss(one_hot, logits, from_logits=False):
        return 1. - tf.reduce_mean(dice_score_from_logits(one_hot, logits, from_logits=from_logits))


def dice_score_from_logits(y_true, y_pred, from_logits=False):
        probs = tf.nn.softmax(y_pred) if from_logits else y_pred
        # Axes which don't contain batches or classes (i.e. exclude first and last axes)
        target_axes = list(range(len(probs.shape)))[1:-1]
        intersect = tf.reduce_sum(probs * y_true, axis=target_axes)
        denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(y_true, axis=target_axes)
        dice_score = tf.reduce_mean(2. * intersect / (denominator + 1e-6), axis=0)
        return dice_score


if __name__ == '__main__':
    x = tf.constant([[
        [1., 0., 0.],
        [0., 1., 0.]
    ]])

    y = tf.constant([[
        [1., 0., 0.],
        [0., 1., 0.]
    ]])
    print(dice_score_from_logits(y, x))