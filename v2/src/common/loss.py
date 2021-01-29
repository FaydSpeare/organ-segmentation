import tensorflow as tf


class Losses:

    @staticmethod
    def get_loss_fn(params):
        name = params['loss_fn']
        if name == 'DICE':
            return dice_loss
        elif name == 'GDICE':
            return generalised_dice_loss
        elif name == 'BDICE':
            return batch_dice_loss
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


def generalised_dice_loss(one_hot, logits, from_logits=False):
    probs = tf.nn.softmax(logits) if from_logits else logits
    # Axes which don't contain batches or classes (i.e. exclude first and last axes)
    target_axes = list(range(len(probs.shape)))[1:-1]
    weights = 1 / (tf.maximum(tf.reduce_sum(one_hot, axis=target_axes), 1e-6) ** 2)
    intersect = tf.reduce_sum(probs * one_hot, axis=target_axes)
    denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(one_hot, axis=target_axes)
    dice_score = tf.reduce_mean(2. * (weights * intersect) / tf.maximum(weights * denominator, 1e-6), axis=0)
    return 1 - tf.reduce_mean(dice_score)


# This appears to produces strange result. Needs to be investigated.
# Not sure whether to use with regular or batch dice loss.
def __dice_loss_missing_classes(one_hot, logits, from_logits=False):
    target_axes = list(range(len(one_hot.shape)))[:-1]
    present_classes = tf.cast(tf.math.count_nonzero(tf.reduce_sum(one_hot, axis=target_axes)), tf.float32)
    return 1. - tf.reduce_sum(dice_score_from_logits(one_hot, logits, from_logits=from_logits)) / present_classes


def dice_loss(one_hot, logits, from_logits=False):
    return 1. - tf.reduce_mean(dice_score_from_logits(one_hot, logits, from_logits=from_logits))


# Functions for regular dice loss
def dice_score_from_logits(y_true, y_pred, from_logits=False):
    probs = tf.nn.softmax(y_pred) if from_logits else y_pred
    # Axes which don't contain batches or classes (i.e. exclude first and last axes)
    target_axes = list(range(len(probs.shape)))[1:-1]
    intersect = tf.reduce_sum(probs * y_true, axis=target_axes)
    denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(y_true, axis=target_axes)
    dice_score = tf.reduce_mean(2. * intersect / (denominator + 1e-6), axis=0)
    return dice_score


## Functions for batch dice loss
def batch_dice_loss(one_hot, logits, from_logits=False):
    return 1. - tf.reduce_mean(batch_dice_score_from_logits(one_hot, logits, from_logits=from_logits))


def batch_dice_score_from_logits(y_true, y_pred, from_logits=False):
    probs = tf.nn.softmax(y_pred) if from_logits else y_pred
    # Axes which don't contain batches or classes (i.e. exclude first and last axes)
    target_axes = list(range(len(probs.shape)))[:-1]
    intersect = tf.reduce_sum(probs * y_true, axis=target_axes)
    denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(y_true, axis=target_axes)
    dice_score = 2. * intersect / (denominator + 1e-6)
    return dice_score


if __name__ == '__main__':
    x = tf.constant([[
        [1., 0., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ]])

    y = tf.constant([[
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.]
    ]])
    print(generalised_dice_loss(y, x))
    print(dice_loss(y, x))
    print(dice_score_from_logits(y, x))