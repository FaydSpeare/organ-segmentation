import tensorflow as tf

from common import TensorBoard, Optimiser
from common.loss import batch_dice_score_from_logits
import common.parameters as p


class Solver:


    def __init__(self, network, params):
        self.network = network
        self.params = params
        self.optimiser = Optimiser.optimiser(params[p.OPTIMISER])(learning_rate=params[p.LR])
        self.loss_fn = params[p.LOSS_FN]
        self.tensorboard = TensorBoard(params)
        self.metrics = self.init_metrics()
        self.epoch = 0
        self.early_stopping_tick = 0
        self.best_val_loss = None


    def init_metrics(self):
        metrics = dict()
        metrics['base_output_loss'] = tf.keras.metrics.Mean(name='base_output_loss')
        metrics['imitation_output_loss'] = tf.keras.metrics.Mean(name='imitation_output_loss')
        metrics['label_output_loss'] = tf.keras.metrics.Mean(name='label_output_loss')
        metrics['imitation_loss'] = tf.keras.metrics.Mean(name='imitation_loss')
        for i in range(self.params[p.NUM_CLASSES]):
            metrics[f'base_dice_score_{i}'] = tf.keras.metrics.Mean(name=f'base_dice_score_{i}')
            metrics[f'imitation_dice_score_{i}'] = tf.keras.metrics.Mean(name=f'imitation_dice_score_{i}')
            metrics[f'label_dice_score_{i}'] = tf.keras.metrics.Mean(name=f'label_dice_score_{i}')
        return metrics


    def update_metrics(self, batch_metrics):
        self.metrics['base_output_loss'].update_state(batch_metrics['base_output_loss'])
        self.metrics['imitation_output_loss'].update_state(batch_metrics['imitation_output_loss'])
        self.metrics['label_output_loss'].update_state(batch_metrics['label_output_loss'])
        self.metrics['imitation_loss'].update_state(batch_metrics['imitation_loss'])
        for i in range(self.params[p.NUM_CLASSES]):
            self.metrics[f'base_dice_score_{i}'].update_state(batch_metrics['base_dice_score'][i])
            self.metrics[f'imitation_dice_score_{i}'].update_state(batch_metrics['imitation_dice_score'][i])
            self.metrics[f'label_dice_score_{i}'].update_state(batch_metrics['label_dice_score'][i])


    def current_metrics(self):
        return {key: metric.result() for key, metric in self.metrics.items()}


    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset_states()


    def run_epoch(self, dataset, mode):
        assert mode in self.params[p.MODES]
        is_training = (mode == 'train')
        for batch in dataset:
            y_label = tf.keras.utils.to_categorical(tf.cast(batch['Y'], tf.int32), num_classes=self.params[p.NUM_CLASSES])
            y_input = tf.keras.utils.to_categorical(tf.cast(batch['Y'], tf.float32), num_classes=self.params[p.NUM_CLASSES])
            loss, logits = self.step([batch['X'], y_input], y_label, training=is_training)
            dice_scores = {name.split('_')[0] + '_dice_score' : batch_dice_score_from_logits(y_label, logits[name]) for name in logits}
            loss.update(dice_scores)
            self.update_metrics(loss)

        if mode == 'val': self.save_model()
        if mode == 'train' : self.epoch += 1
        self.tensorboard.write_scalars(self.metrics, mode, self.epoch)
        current_metrics = self.current_metrics()
        self.reset_metrics()
        return current_metrics



    # @tf.function
    def step(self, x, y, training):
        # print('step')
        if training:
            with tf.GradientTape(persistent=True) as tape:
                [base_d_out, im_d_out, label_d_out], [im_e_out, label_e_out] = self.network(x, training=True)

                base_output_loss = self.loss_fn(y, base_d_out)
                imitation_output_loss = self.loss_fn(y, im_d_out)
                label_output_loss = self.loss_fn(y, label_d_out)
                imitation_loss = tf.norm(im_e_out - label_e_out, ord='euclidean')# / tf.size(im_e_out, out_type=tf.float32)

            # Label Output Loss
            # print('label')
            label_output_trainables = self.network.label_encoder.trainable_variables + self.network.label_decoder.trainable_variables
            gradients = tape.gradient(label_output_loss, label_output_trainables)
            self.optimiser.apply_gradients(zip(gradients, label_output_trainables))

            # Imitation Output Loss
            # print('im2')
            imitation_output_trainables = self.network.imitating_encoder.trainable_variables + self.network.label_decoder.trainable_variables
            gradients = tape.gradient(imitation_output_loss, imitation_output_trainables)
            self.optimiser.apply_gradients(zip(gradients, imitation_output_trainables))

            # Imitation Loss
            # print('im')
            gradients = tape.gradient(imitation_loss, self.network.imitating_encoder.trainable_variables)
            self.optimiser.apply_gradients(zip(gradients, self.network.imitating_encoder.trainable_variables))

            # Base Output Loss
            # print('base')
            base_output_trainables = self.network.base_encoder.trainable_variables + self.network.base_decoder.trainable_variables
            gradients = tape.gradient(base_output_loss, base_output_trainables)
            self.optimiser.apply_gradients(zip(gradients, base_output_trainables))

        else:
            [base_d_out, im_d_out, label_d_out], [im_e_out, label_e_out] = self.network(x, training=False)

            base_output_loss = self.loss_fn(y, base_d_out)
            imitation_output_loss = self.loss_fn(y, im_d_out)
            label_output_loss = self.loss_fn(y, label_d_out)
            imitation_loss = tf.norm(im_e_out - label_e_out, ord='euclidean')# / tf.size(im_e_out, out_type=tf.float32)

        loss = {
            # Loss
            'base_output_loss': base_output_loss,
            'imitation_output_loss': imitation_output_loss,
            'label_output_loss': label_output_loss,
            'imitation_loss': imitation_loss
        }

        logits = {
            'base_output_logits': base_d_out,
            'imitation_output_logits': im_d_out,
            'label_output_logits': label_d_out
        }

        return loss, logits


    def save_model(self):
        smaller_loss = self.best_val_loss is None or self.metrics['imitation_output_loss'].result() < self.best_val_loss
        if smaller_loss:
            self.early_stopping_tick = 0
            self.best_val_loss = self.metrics['imitation_output_loss'].result()
            self.network.save_weights(self.params[p.MODEL_PATH] + '/model_weights')
            print("Saving new model.", flush=True)
        self.early_stopping_tick += 1

