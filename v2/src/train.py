from common.tfrecords import TFRecordsManager
from model.solver import Solver
from model.network import CDFNet, ViewAggregation
from common import misc, Optimiser, Loss
import common.parameters as p
from predict import predict

def main():

    # Parameters for training
    parameters = p.default_parameters()
    parameters[p.PREFIX]       = 'HM_50_COMBOL'
    parameters[p.TFRECORDS]    = 'histmatch/axial'
    parameters[p.LOSS_FN]      =  Loss.CCE_BDICE
    parameters[p.LR]           =  0.001
    parameters[p.NUM_CLASSES]  =  5
    parameters[p.OPTIMISER]    =  Optimiser.ADAM
    parameters[p.TRAIN_BATCH]  =  27
    parameters[p.VAL_BATCH]    =  40
    parameters[p.PATIENCE]     =  20
    parameters[p.NETWORK]      =  CDFNet
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
        val_loss = epoch_metrics['val']['loss']
        print(f'ValLoss:[{val_loss}] BestValLoss:[{best_val_loss}] EST:[{solver.early_stopping_tick}]', flush=True)
        if solver.early_stopping_tick > parameters[p.PATIENCE]:
            break

    if type(parameters[p.NETWORK]()) == CDFNet:
        # Run predictions for dataset
        model_folder = parameters[p.MODEL_PATH].split('/')[-2]
        predict(model_folder, parameters[p.TFRECORDS], prefix=parameters[p.PREFIX])

if __name__ == '__main__':
    main()