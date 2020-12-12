
from tf_utils import TFRecordsManager, misc, CDFNet
from model.solver import SegSolver

def main():

    params = {
        'loss_fn' : 'DICEL',
        'out_channels' : 5,
        'learning_rate' : 0.001
    }

    base_path, model_path = misc.get_base_path(training=True)
    tfrecords_path = base_path + 'tfrecords/'

    # Load TFRecords
    tfrm = TFRecordsManager()
    dataset = tfrm.load_datasets_without_batching(tfrecords_path)

    # Create Padded Batches
    padding_shapes = {'X': (304, 304, 2), 'Y': (304, 304)}
    for mode in dataset:
        dataset[mode] = dataset[mode].padded_batch(5, padded_shapes=padding_shapes)

    network = CDFNet(num_filters=64, num_classes=5)
    solver = SegSolver(model_path, params, network)

    for epoch in range(100):

        for mode in dataset:
            solver.run_epoch(dataset[mode])



if __name__ == '__main__':
    main()