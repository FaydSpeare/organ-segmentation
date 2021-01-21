import tensorflow as tf


class TensorBoard:


    def __init__(self, params):
        self.path = params['path']
        self.modes = params['modes']
        self.writers = self.create_summary_writers()


    def create_summary_writers(self):
        return {mode: tf.summary.create_file_writer(self.path + '/logs/' + mode) for mode in self.modes}


    def write_scalars(self, metrics, mode, step):
        with self.writers[mode].as_default():
            for metric in metrics.values():
                tf.summary.scalar(metric.name, metric.result(), step=step)


    def write_images(self, images, name, mode, step, max_images=3):
        with self.writers[mode].as_default():
            tf.summary.image(name, images, max_outputs=max_images, step=step)
