import tensorflow as tf
import numpy as np

from tf_utils import Tensorboard, MetricsManager, misc

class SegSolver:

    def __init__(self, model_path, params, network):
        self.model_path = model_path
        self.params = params
        self.network = network
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        self.modes = ['train', 'val']

        self.metrics = self.init_metrics()
        self.mm = MetricsManager()
        self.tb = Tensorboard(model_path, self.metrics['train'], self.modes)

        self.early_stopping_tick = 0

    def init_metrics(self):
        return {
            mode : {
                'loss' : {'value' : 10000., 'type' : 'Mean'},
                'score_by_class' : [{'value' : 0., 'type' : 'Mean'} for _ in range(self.params['out_channels'])],
                'accuracy_by_class' : [{'value' : 0., 'type' : 'Mean'} for _ in range(self.params['out_channels'])]
            }
            for mode in self.modes
        }

    def run_epoch(self, dataset, mode, epoch):

        assert mode in ['train', 'val']

        for batch in dataset:
            y = tf.keras.utils.to_categorical(tf.cast(batch['Y'], tf.int32), num_classes=self.params['out_channels'])
            predictions, metrics = self.step(batch['X'], y, training=(mode == 'train'))
            self.tb.update_metrics(metrics)

        if mode == 'val':
            self.save_model(mode)


        self.tb.write_summary(mode, epoch)

    #@tf.function
    def step(self, x, y, training=True):

        loss_fn = self.mm.metrics[self.params['loss_fn']]

        if training:

            with tf.GradientTape() as tape:
                logits = self.network(x, training=True)
                loss = loss_fn(y, logits, probs=True)

            gradients = tape.gradient(loss, self.network.trainable_variables)
            scores_by_class = self.mm.dice_score_from_logits(y, logits, probs=True)
            class_accuracies = self.class_accuracy_from_logits(y, logits)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        else:

            logits = self.network(x, training=False)
            scores_by_class = self.mm.dice_score_from_logits(y, logits, probs=True)
            class_accuracies = self.class_accuracy_from_logits(y, logits)
            loss = loss_fn(y, logits, probs=True)

        return misc.get_argmax_prediction(logits), {'loss' : loss, 'score_by_class' : scores_by_class, 'accuracy_by_class' : class_accuracies}




    def class_accuracy_from_logits(self, one_hot, probs):

        # Axes which don't contain batches or classes (i.e. exclude first and last axes)
        target_axes = list(range(len(probs.shape)))[1:-1]

        argmax_one_hot = tf.keras.utils.to_categorical(tf.math.argmax(probs, axis=-1), num_classes=self.params['out_channels'])
        intersect = tf.reduce_sum(argmax_one_hot * one_hot, axis=target_axes)
        denominator = tf.reduce_sum(one_hot, axis=target_axes)

        class_accuracies = tf.reduce_mean(intersect / (denominator + 1e-6), axis=0)

        return class_accuracies

    def save_model(self, mode):
        epoch_metrics = self.tb.get_current_metrics()
        smaller_loss = epoch_metrics['loss'] < self.metrics[mode]['loss']['value']
        if smaller_loss:
            self.early_stopping_tick = 0
            self.metrics[mode]['loss']['value'] = epoch_metrics['loss']
            base_path, _ = misc.get_base_path(training=True)
            self.network.save_weights(base_path + '/model')
        self.early_stopping_tick += 1








