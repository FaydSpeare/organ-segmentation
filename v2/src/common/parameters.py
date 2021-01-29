from common import Loss, Optimiser

PREFIX = 'prefix'
TFRECORDS = 'tfrecords'
LOSS_FN = 'loss_fn'
NUM_CLASSES = 'out_channels'
MODES = 'modes',
MODEL_PATH = 'path'
OPTIMISER = 'optimiser'
LR = 'learning_rate'
TRAIN_BATCH = 'batch_size'
VAL_BATCH = 'val_batch_size'
PATIENCE = 'patience'

def default_parameters():
    return {
        PREFIX : '',
        LOSS_FN : Loss.DICE,
        LR : 0.001,
        OPTIMISER : Optimiser.ADAM,
        MODES : ['train', 'val'],
        TRAIN_BATCH : 10,
        VAL_BATCH : 10,
        PATIENCE : 10
    }

def validate(p):
    try:
        assert PREFIX in p and type(p[PREFIX]) == str
        assert LOSS_FN in p and type(p[LOSS_FN]) == Loss
        assert NUM_CLASSES in p and type(p[NUM_CLASSES]) == int
        assert NUM_CLASSES in p and type(p[NUM_CLASSES]) == int
        assert LR in p and type(p[LR]) == float
        assert OPTIMISER in p and type(p[OPTIMISER]) == Optimiser
        assert MODES in p and type(p[MODES]) == list
        assert all(type(mode) == str for mode in p[MODES])
        assert TRAIN_BATCH in p and type(p[TRAIN_BATCH]) == int
        assert VAL_BATCH in p and type(p[VAL_BATCH]) == int
        assert PATIENCE in p and type(p[PATIENCE]) == int
    except AssertionError:
        return False
    return True
