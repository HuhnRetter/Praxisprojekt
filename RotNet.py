import torch
import torch.nn as nn
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter

from HelperFunctions.DataFunctions.DataHelperFunctions import dataloaderSetup
# tensorboard --logdir=runs
###################################################
from HelperFunctions.DataFunctions.EnumsForParameter import LR_Scheduler, TRAINING_MODE, Optimizer
from HelperFunctions.Extras.NameGeneratorFunctions import nameGenerator
from HelperFunctions.General.ModelSetupFunctions import rotNetSetup
from HelperFunctions.General.SetupHelper import OptimizerAndSchedulerSetup, TrainingModeSetup

device = torch.device('dml')
confusionmatrixdevice = torch.device('dml')

# Output Settings
all_classes = ["rot0", "rot90", "rot180", "rot270"]
confusionmatrixeveryXepoch = 3
# Data Settings
DATASETPATH = './datasets/'
all_datasetclasses_train = ["Dog"]
all_transforms_train = ['rot0', 'rot90', 'rot180', 'rot270']
all_datasetclasses_test = ["Dog", "Other", "Flower"]
all_transforms_test = ['rot0', 'rot90', 'rot180', 'rot270']
num_workers = 2
pin_memory = True
# Model Settings
MODELFOLDER = './Models/'
# Train Parameter
num_classes = len(all_classes)
num_epochs = 8
batch_size = 4
learning_rate = 0.0001
weight_decay = 0
optimizer_param = Optimizer.ADAM
scheduler_mode = LR_Scheduler.NONE
training_mode = TRAINING_MODE.NORMAL
# SGD optimizer
momentum = 0.9
# ReduceLROnPlateau Scheduler
patience = 1
# CosineAnnealingWarmRestart Scheduler
T_0 = 1
# CycleLR Scheduler
dataset_size = 41600 * len(all_datasetclasses_train)
base_lr = 0
max_lr = 0.02
cycle_epochs = 8

# 1 == True ; 0 == False
save_trained_model = 1

# Automatic Filename for  saving
SAVEFILE = nameGenerator(all_datasetclasses_train, num_epochs, batch_size, learning_rate, weight_decay, optimizer_param,
                         scheduler_mode, momentum, MODELFOLDER, T_0, False, customcomment="RotNetTraining")

# Manuel Filename for loading
LOADFILE = "TestDogDataBS4.pth"


# Writer for Tensorboard
Tensorboard_name = nameGenerator(all_datasetclasses_train, num_epochs, batch_size, learning_rate, weight_decay,
                                 optimizer_param, scheduler_mode, momentum, MODELFOLDER, T_0, True,
                                 customcomment="")
writer = SummaryWriter(f'Tensorboard/runs/{Tensorboard_name}')


def main():
    model = rotNetSetup(num_classes).to(device)
    train_loader, test_loader = dataloaderSetup(DATASETPATH, all_datasetclasses_train, all_transforms_train,
                                                all_datasetclasses_test, all_transforms_test, batch_size, num_workers,
                                                pin_memory)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = OptimizerAndSchedulerSetup(optimizer_param, scheduler_mode, model, learning_rate, weight_decay, momentum, T_0, patience)

    if save_trained_model == 0:
        print(f'WARNING! Model will not be saved!\n\n')

    TrainingModeSetup(training_mode, model, criterion, optimizer, scheduler, train_loader, num_epochs,
                      save_trained_model,
                      device,
                      confusionmatrixdevice, writer, SAVEFILE, all_classes, confusionmatrixeveryXepoch,
                      scheduler_mode, test_loader, LOADFILE, base_lr, max_lr, dataset_size, batch_size, cycle_epochs)


if __name__ == "__main__":
    main()
