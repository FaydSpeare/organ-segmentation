import tensorflow as tf
import os

from . import misc


class TFRecordsManager:

    def __init__(self):
        self.types = {
            "float32": tf.float32,
            "int32": tf.int32
        }

    @staticmethod
    def _bytes_feature(value):
        """ Returns a bytes_list from a string / byte. """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """ Returns a float_list from a float / double. """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """ Returns an int64_list from a bool / enum / int / uint. """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _serialize_tensor(x):
        """ Serializes a tensor """
        return TFRecordsManager._bytes_feature(tf.io.serialize_tensor(tf.constant(x)))

    def save_record(self, path, data):
        """ Creates a TFRecord from the data """
        writer = tf.io.TFRecordWriter(path + ".tfrecord", options=tf.io.TFRecordOptions(compression_type="GZIP"))

        # Iterate over each sample
        for i in range(len(data)):
            # Iterate over keys of data_i (X, Y, etc.) and serialize
            features = {key: self._serialize_tensor(data[i][key]) for key in data[i]}

            # Write example
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

        writer.close()

    @staticmethod
    def get_record_filenames(path):
        return sorted(path + filename for filename in os.listdir(path))

    def parse_TFRecord(self, record, keys):
        features = {key: tf.io.FixedLenFeature([], tf.string) for key in keys}
        parsed_record = tf.io.parse_single_example(record, features)
        return {key: tf.io.parse_tensor(parsed_record[key], self.types[keys[key]]) for key in keys}

    def load_datasets(self, path, batch_size):
        params = misc.load_json(path + "params.json")

        datasets = {}
        for data_purpose in params["data_purposes"]:
            records = TFRecordsManager.get_record_filenames(path + data_purpose + "/")
            dataset = tf.data.TFRecordDataset(records, compression_type='GZIP').map(lambda record: self.parse_TFRecord(record, params['data_keys']))

            dataset = dataset.shuffle(250)
            dataset = dataset.batch(batch_size) if "train" in data_purpose else dataset.batch(1)
            datasets[data_purpose] = dataset
            del dataset

        return datasets

    def load_datasets_without_batching(self, path):
        params = misc.load_json(path + "params.json")

        datasets = {}
        for data_purpose in params["data_purposes"]:
            records = TFRecordsManager.get_record_filenames(path + data_purpose + "/")
            dataset = tf.data.TFRecordDataset(records, compression_type='GZIP').map(lambda record: self.parse_TFRecord(record, params['data_keys']))

            dataset = dataset.shuffle(100)
            datasets[data_purpose] = dataset
            del dataset

        return datasets
