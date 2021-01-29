import tensorflow as tf


class MetricsManager:

    def __init__(self):
        self.metrics = {
            "GDICEL": self.generalize_dice_loss,
            "DICEL": self.dice_loss,
            "CE": self.cross_entropy,
            "WCE": self.weighted_cross_entropy,
            "L1": self.L1_loss,
            "L2": self.L2_loss,
            "RMSE": self.RMSE,
            "SSIM": self.SSIM
        }

        self.losses = {
            "L1": tf.keras.losses.MeanAbsoluteError(),
            "L2": tf.keras.losses.MeanSquaredError()
        }

    def generalize_dice_loss(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[1, 2, 3])
        w = 1 / (w ** 2 + 1e-6)
        # w = w / tf.reduce_sum(w)  # Normalize weights

        probs = tf.nn.softmax(logits)

        multed = tf.reduce_sum(probs * one_hot, axis=[1, 2, 3])
        summed = tf.reduce_sum(probs, axis=[1, 2, 3]) + tf.reduce_sum(one_hot, axis=[1, 2, 3])

        numerator = w * multed
        denominator = w * summed

        dice_score = tf.reduce_mean(2. * numerator / (denominator + 1e-6), axis=0)

        return 1. - tf.reduce_mean(dice_score)

    def dice_score_from_logits(self, one_hot, logits, probs=False):
        """
        Dice coefficient (F1 score) is between 0 and 1.
        :param labels: one hot encoding of target (num_samples, num_classes)
        :param logits: output of network (num_samples, num_classes)
        :return: Dice score by each class
        """

        probs = tf.nn.softmax(logits) if not probs else logits

        # Axes which don't contain batches or classes (i.e. exclude first and last axes)
        target_axes = list(range(len(probs.shape)))[1:-1]

        intersect = tf.reduce_sum(probs * one_hot, axis=target_axes)
        denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(one_hot, axis=target_axes)

        dice_score = tf.reduce_mean(2. * intersect / (denominator + 1e-6), axis=0)

        return dice_score

    def dice_loss(self, one_hot, logits, probs=False):
        return 1. - tf.reduce_mean(self.dice_score_from_logits(one_hot, logits, probs=probs))

    def cross_entropy(self, onehot, logits, probs=False):
        ce = tf.losses.categorical_crossentropy(onehot, logits, from_logits=(not probs))
        #ce = tf.nn.softmax_cross_entropy_with_logits(onehot, logits, axis=-1)
        return tf.reduce_mean(ce)

    def weighted_cross_entropy(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[0, 1, 2, 3])
        w = 1 / (w + 1e-6)
        w = w / tf.reduce_sum(w)  # Normalize weights
        # w = tf.constant([0.01, 3., 2., 2., 20., 3., 20., 20., 4., 5.])

        ce = tf.nn.softmax_cross_entropy_with_logits(one_hot, logits, axis=-1)
        weights = tf.reduce_sum(w * one_hot, axis=-1)

        return tf.reduce_mean(weights * ce)

    def L1_loss(self, labels, logits):
        return self.losses["L1"](labels, logits)

    def L2_loss(self, labels, logits):
        return self.losses["L2"](labels, logits)

    def RMSE(self, labels, logits, mask):
        if mask is None:
            mask = tf.ones_like(labels)

        true_flat = tf.keras.layers.Flatten()(labels)
        fake_flat = tf.keras.layers.Flatten()(logits)
        mask_flat = tf.keras.layers.Flatten()(mask)

        # Get only elements in mask
        true_new = tf.boolean_mask(true_flat, mask_flat)
        fake_new = tf.boolean_mask(fake_flat, mask_flat)

        # Demean
        true_demean = true_new - tf.math.reduce_mean(true_new)
        fake_demean = fake_new - tf.math.reduce_mean(fake_new)

        return 100 * tf.norm(true_demean - fake_demean) / tf.norm(true_demean)

    def SSIM(self, img1, img2, window, K1=0.01, K2=0.03, padding="VALID"):

        mu1 = tf.nn.conv3d(img1, window, strides=[1, 1, 1, 1, 1], padding=padding)
        mu2 = tf.nn.conv3d(img2, window, strides=[1, 1, 1, 1, 1], padding=padding)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = tf.nn.conv3d(img1 * img1, window, strides=[1, 1, 1, 1, 1], padding=padding) - mu1_sq
        sigma2_sq = tf.nn.conv3d(img2 * img2, window, strides=[1, 1, 1, 1, 1], padding=padding) - mu2_sq
        sigma12 = tf.nn.conv3d(img1 * img2, window, strides=[1, 1, 1, 1, 1], padding=padding) - mu1_mu2

        C1 = K1 ** 2
        C2 = K2 ** 2

        return tf.reduce_mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
