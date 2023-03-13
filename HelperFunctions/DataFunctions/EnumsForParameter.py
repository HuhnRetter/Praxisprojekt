from enum import Enum


class Optimizer(Enum):
    ADAM = "Adam"
    SGD = "SGD"


class LR_Scheduler(Enum):
    REDUCELR0NPLATEAU = "ReduceLROnPlateau"
    COSINEANNEALINGWARMRESTARTS = "CosineAnnealingWarmRestarts"
    NONE = "none"


class TRAINING_MODE(Enum):
    ESTIMATE_LR = "EstimateLR"
    NORMAL = "Normal"
    LOAD_EXISTING_MODEL_TRAINING = "LOAD_EXISTING_MODEL_TRAINING"
    LOAD_EXISTING_MODEL_TESTING = "LOAD_EXISTING_MODEL_TESTING"


class RANGE_PERMUT_MODE(Enum):
    NORMAL = "Normal"
    NONE = "none"


class RANGE_INDICES_MODE(Enum):
    INT = "Int"
    FLOAT = "Float"


class DATASET_CHOSEN(Enum):
    IMAGEDATASET = "Imagedataset"
    JIGSAWDATASET = "Jigsawdataset"


class TEST_MODE(Enum):
    CONFUSION_MATRIX = "Confusionmatrix"
    AVG_ACC = "Avg_Acc"


class PRETRAINED_MODEL(Enum):
    ROTNET = "Rotnet"
    JIGSAW = "Jigsaw"
    RESNET18 = "Resnet18"
    RANDOM = "Random"
