import tensorflow as tf

from common import TensorBoard
from common.loss import batch_dice_score_from_logits
import common.parameters as p


class Solver:


    def __init__(self, network, params):
        self.network = network
        self.params = params
        self.optimiser = params[p.OPTIMISER](learning_rate=params[p.LR])
        self.loss_fn = params[p.LOSS_FN]
        self.tensorboard = TensorBoard(params)
        self.metrics = self.init_metrics()
        self.epoch = 0
        self.early_stopping_tick = 0
        self.best_val_loss = None


    def init_metrics(self):
        metrics = {'loss': tf.keras.metrics.Mean(name='loss')}
        for i in range(self.params[p.NUM_CLASSES]):
            metrics[f'dice_score_{i}'] = tf.keras.metrics.Mean(name=f'dice_score_{i}')
            metrics[f'accuracy_{i}'] = tf.keras.metrics.Mean(name=f'accuracy_{i}')
        return metrics


    def update_metrics(self, batch_metrics):
        self.metrics['loss'].update_state(batch_metrics['loss'])
        for i in range(self.params[p.NUM_CLASSES]):
            self.metrics[f'dice_score_{i}'].update_state(batch_metrics['dice_scores'][i])
            self.metrics[f'accuracy_{i}'].update_state(batch_metrics['accuracies'][i])


    def current_metrics(self):
        return {key: metric.result() for key, metric in self.metrics.items()}


    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset_states()


    def run_epoch(self, dataset, mode):
        assert mode in self.params[p.MODES]
        is_training = mode == 'train'
        for batch in dataset:
            y = tf.keras.utils.to_categorical(tf.cast(batch['Y'], tf.int32), num_classes=self.params[p.NUM_CLASSES])
            logits, loss = self.step(batch['X'], y, training=is_training)

            # Update metrics
            dice_scores = batch_dice_score_from_logits(y, logits)
            class_accuracies = self.class_accuracy_from_logits(y, logits)
            self.update_metrics({'loss' : loss, 'dice_scores' : dice_scores, 'accuracies' : class_accuracies})

        if mode == 'val': self.save_model()
        if mode == 'train' : self.epoch += 1
        self.tensorboard.write_scalars(self.metrics, mode, self.epoch)
        current_metrics = self.current_metrics()
        self.reset_metrics()
        return current_metrics



    # @tf.function
    def step(self, x, y, training=True):

        if training:
            with tf.GradientTape() as tape:
                logits = self.network(x, training=True)
                loss = self.loss_fn(y, logits)
            gradients = tape.gradient(loss, self.network.trainable_variables)
            self.optimiser.apply_gradients(zip(gradients, self.network.trainable_variables))

        else:
            logits = self.network(x, training=False)
            loss = self.loss_fn(y, logits)

        return logits, loss




    def class_accuracy_from_logits(self, one_hot, probs):

        # Axes which don't contain batches or classes (i.e. exclude first and last axes)
        target_axes = list(range(len(probs.shape)))[0:-1]

        argmax_one_hot = tf.keras.utils.to_categorical(tf.math.argmax(probs, axis=-1), num_classes=self.params[p.NUM_CLASSES])
        intersect = tf.reduce_sum(argmax_one_hot * one_hot, axis=target_axes)
        denominator = tf.reduce_sum(one_hot, axis=target_axes)

        class_accuracies = intersect / (denominator + 1e-6)

        return class_accuracies


    def save_model(self):
        smaller_loss = self.best_val_loss is None or self.metrics['loss'].result() < self.best_val_loss
        if smaller_loss:
            self.early_stopping_tick = 0
            self.best_val_loss = self.metrics['loss'].result()
            self.network.save_weights(self.params[p.MODEL_PATH] + '/model_weights')
        self.early_stopping_tick += 1

