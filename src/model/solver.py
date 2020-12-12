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

    def init_metrics(self):
        return {
            mode : {
                'loss' : {'value' : 999., 'type' : 'Mean'},
                'scores_by_class' : [{'value' : 0., 'type' : 'Mean'} for _ in range(self.params['out_channels'])]
            }
            for mode in self.modes
        }

    def run_epoch(self, dataset, mode, epoch):

        for batch in dataset:
            y = tf.keras.utils.to_categorical(tf.cast(batch['Y'], tf.int32), num_classes=self.params['out_channels'])
            training = True if mode == 'train' else False
            predictions, metrics = self.step(batch['X'], y, training=training)
            self.tb.update_metrics(metrics)

        self.tb.write_summary(mode, epoch)

    @tf.function
    def step(self, x, y, training=True):

        loss_fn = self.mm.metrics[self.params['loss_fn']]

        if training:

            with tf.GradientTape() as tape:
                logits = self.network(x, training=True)
                loss = loss_fn(y, logits)

            gradients = tape.gradient(loss, self.network.trainable_variables)
            scores_by_class = self.mm.dice_score_from_logits(y, logits)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        else:

            logits = self.network(x, training=False)
            scores_by_class = self.mm.dice_score_from_logits(y, logits)
            loss = loss_fn(y, logits)

        return misc.get_argmax_prediction(logits), {'loss' : loss, 'scores_by_class' : scores_by_class}













