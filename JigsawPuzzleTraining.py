############## TENSORBOARD ########################
import os

import torch
import torch.nn as nn
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter

from HelperFunctions.DataFunctions.DataHelperFunctions import dataloaderSetup
# tensorboard --logdir=runs
###################################################
from HelperFunctions.DataFunctions.EnumsForParameter import LR_Scheduler, TRAINING_MODE, Optimizer, \
    TEST_MODE, RANGE_PERMUT_MODE, DATASET_CHOSEN
from HelperFunctions.Extras.NameGeneratorFunctions import nameGenerator
from HelperFunctions.Extras.PermutationFunctions import buildPermutation
from HelperFunctions.General.ModelSetupFunctions import JigsawNetwork
from HelperFunctions.General.SetupHelper import OptimizerAndSchedulerSetup, TrainingModeSetup

# tensorboard --logdir=runs
###################################################

device = torch.device('dml')
confusionmatrixdevice = torch.device('dml')

amount_of_top_permuts = 100
amount_of_permuts = 69
# Output Settings
all_classes = list()
for i in range(0, amount_of_top_permuts):
    all_classes.append(f"{i}")
confusionmatrixeveryXepoch = 100
test_mode = TEST_MODE.AVG_ACC
# Data Settings
DATASETPATH = './datasets/'
all_datasetclasses_train = ["Dog"] # ein dataset is 10400 groß * anzahl der permutation
all_transforms_train = ['jigsaw']
all_datasetclasses_test = ["Dog", "Other", "Flower"]
all_transforms_test = ['jigsaw']
num_workers = 2
pin_memory = True
# Model Settings
MODELFOLDER = './Models/'
permuts_FILEPATH = f"./top_{amount_of_top_permuts}_permuts.npy"
range_permut_mode = RANGE_PERMUT_MODE.NORMAL
dataset_choosen = DATASET_CHOSEN.JIGSAWDATASET
siamese_deg = 9
# Train Parameter
num_classes = len(all_classes)
num_epochs = 4
batch_size = 150 #bei 130/160 n_every_steps = 80
learning_rate = 0.01
weight_decay = 0
optimizer_param = Optimizer.SGD
scheduler_mode = LR_Scheduler.NONE
training_mode = TRAINING_MODE.LOAD_EXISTING_MODEL_TESTING
# SGD optimizer
momentum = 0.9
# ReduceLROnPlateau Scheduler
patience = 1
# CosineAnnealingWarmRestart Scheduler
T_0 = 1
# CycleLR Scheduler
dataset_size = siamese_deg * 10400 * len(all_datasetclasses_train)
base_lr = 0
max_lr = 0.02
cycle_epochs = 8



# 1 == True ; 0 == False
save_trained_model = 1

# Automatic Filename for  saving
SAVEFILE = nameGenerator(all_datasetclasses_train, num_epochs, batch_size, learning_rate, weight_decay, optimizer_param,
                         scheduler_mode, momentum, MODELFOLDER, T_0, False, customcomment="JigsawTraining")

# Manuel Filename for loading
LOADFILE = "./JigsawModels/JigsawAcc89Dog.pth"


# Writer for Tensorboard
Tensorboard_name = nameGenerator(all_datasetclasses_train, num_epochs, batch_size, learning_rate, weight_decay,
                                 optimizer_param, scheduler_mode, momentum, MODELFOLDER, T_0, True,
                                 customcomment=f"_AmountPermuts{amount_of_permuts}")




def main():
    if os.path.exists(f'Tensorboard/runs/Jigsaw/{Tensorboard_name}') and training_mode == TRAINING_MODE.NORMAL \
            or training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TRAINING:
        raise Exception("WARNING! Overwriting FILE! To continue you need to change the customcomment!\n\n")

    if training_mode == TRAINING_MODE.ESTIMATE_LR or training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TESTING:
        writer = None
    else:
        writer = SummaryWriter(f'Tensorboard/runs/Jigsaw/{Tensorboard_name}')

    if not os.path.exists(permuts_FILEPATH):
        print("Creating Permutation File\n\n")
        buildPermutation(amount_of_top_permuts)

    model = JigsawNetwork(num_classes, siamese_deg).to(device)
    train_loader, test_loader = dataloaderSetup(DATASETPATH, all_datasetclasses_train, all_transforms_train,
                                                all_datasetclasses_test, all_transforms_test, batch_size, num_workers,
                                                pin_memory, dataset_choosen, permuts_FILEPATH, range_permut_mode, amount_of_permuts)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = OptimizerAndSchedulerSetup(optimizer_param, scheduler_mode, model, learning_rate,
                                                      weight_decay, momentum, T_0, patience)

    if save_trained_model == 0:
        print(f'WARNING! Model will not be saved!\n\n')

    TrainingModeSetup(training_mode, model, criterion, optimizer, scheduler, train_loader, num_epochs,
                      save_trained_model,
                      device,
                      confusionmatrixdevice, writer, SAVEFILE, all_classes, confusionmatrixeveryXepoch,
                      scheduler_mode, test_loader, LOADFILE, base_lr, max_lr, dataset_size, batch_size, cycle_epochs,
                      test_mode=test_mode)

if __name__ == "__main__":
    main()