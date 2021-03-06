from common.tfrecords import TFRecordsManager
from model.solver import Solver
from common import misc, Optimiser, Loss
import common.parameters as p
from model.network.network import Network

def main():

    # Parameters for training
    parameters = p.default_parameters()
    parameters[p.PREFIX]       = 'imitation_bn_false'
    parameters[p.TFRECORDS]    = 'hm/axial'
    parameters[p.LOSS_FN]      =  Loss.BDICE
    parameters[p.LR]           =  0.001
    parameters[p.NUM_CLASSES]  =  5
    parameters[p.OPTIMISER]    =  Optimiser.ADAM
    parameters[p.TRAIN_BATCH]  =  16
    parameters[p.VAL_BATCH]    =  16
    parameters[p.PATIENCE]     =  50
    parameters[p.NETWORK]      =  Network
    # parameters[p.NETWORK]      =  ViewAggregation
    p.validate(parameters)

    # Create folder for the new model
    parameters[p.MODEL_PATH] = misc.new_checkpoint_path(prefix=parameters[p.PREFIX], tfr=parameters[p.TFRECORDS])

    # Load TFRecords
    tfrm = TFRecordsManager()
    tfrecord_path = misc.get_tfrecords_path() + f"/{parameters[p.TFRECORDS]}/"
    dataset = tfrm.load_datasets(tfrecord_path, parameters[p.TRAIN_BATCH], parameters[p.VAL_BATCH])

    network = parameters[p.NETWORK](num_classes=parameters[p.NUM_CLASSES])
    solver = Solver(network, parameters)
    epoch_metrics = dict()

    for epoch in range(1000):

        for mode in dataset:
            epoch_metrics[mode] = solver.run_epoch(dataset[mode], mode)

        best_val_loss = solver.best_val_loss
        val_loss = epoch_metrics['val']['imitation_output_loss']
        print(f'ValLoss:[{val_loss}] BestValLoss:[{best_val_loss}] EST:[{solver.early_stopping_tick}]', flush=True)

        for name in ['base', 'imitation', 'label']:
            amount = epoch_metrics["val"][f'{name}_output_loss']
            print(f'{name}_output_loss:', amount, flush=True)

        print(f'imitation_loss:', epoch_metrics["val"]['imitation_loss'], flush=True)
        print()

        if solver.early_stopping_tick > parameters[p.PATIENCE]:
            break

if __name__ == '__main__':
    main()