import tensorflow as tf


class Tensorboard:

    def __init__(self, ckp_path, scalars, modes):
        self.ckp_path = ckp_path
        self.scalars = self.init_metrics(scalars)

        if ckp_path is not None and modes is not None:
            self.summary_writers = {mode: tf.summary.create_file_writer(self.ckp_path + 'logs/' + mode) for mode in modes}

    def init_metrics(self, metrics):
        """
        :param metrics: Dict(stat type => dict(value : 0., type : "Mean"))
        :return:
        """
        scalars = {}
        for key in metrics:
            if isinstance(metrics[key], list):
                scalars[key] = []
                for index in range(len(metrics[key])):
                    if metrics[key][index]["type"] == "Mean":
                        scalars[key].append(tf.keras.metrics.Mean(name=key + '-' + str(index)))
                    else:
                        raise NotImplementedError(metrics[key][index]["type"])
            else:
                if metrics[key]["type"] == "Mean":
                    scalars[key] = tf.keras.metrics.Mean(name=key)
                else:
                    raise NotImplementedError(metrics[key][index]["type"])

        return scalars

    def update_metrics(self, metrics):
        for key in self.scalars:
            if isinstance(self.scalars[key], list):
                for index in range(len(self.scalars[key])):
                    self.scalars[key][index](metrics[key][index])
            else:
                self.scalars[key](metrics[key])

    def get_current_metrics(self):
        metrics = {}
        for key in self.scalars:
            if isinstance(self.scalars[key], list):
                metrics[key] = []
                for index in range(len(self.scalars[key])):
                    metrics[key].append(self.scalars[key][index].result().numpy())
            else:
                metrics[key] = self.scalars[key].result().numpy()

        return metrics

    def write_summary(self, mode, epoch, images=None):
        """ Write statistics. """
        with self.summary_writers[mode].as_default():
            for key in self.scalars:
                # Check if list or not
                if isinstance(self.scalars[key], list):
                    for index in range(len(self.scalars[key])):
                        tf.summary.scalar(key + '-' + str(index), self.scalars[key][index].result(), step=epoch)
                else:
                    tf.summary.scalar(key, self.scalars[key].result(), step=epoch)

            if images is not None:
                for key in list(images.keys()):
                    tf.summary.image(key, images[key][:, :, :, int(images[key].shape[-2] // 2)], max_outputs=2, step=epoch)

        self.reset_states()

    def reset_states(self):
        """ After each epoch reset accumulating metrics. """
        for key in self.scalars:
            if isinstance(self.scalars[key], list):
                for step in range(len(self.scalars[key])):
                    self.scalars[key][step].reset_states()
            else:
                self.scalars[key].reset_states()
