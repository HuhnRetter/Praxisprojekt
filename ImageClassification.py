import os
import torch
import torch.nn as nn
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
###################################################
from HelperFunctions.DataFunctions.EnumsForParameter import PRETRAINED_MODEL, LR_Scheduler, TRAINING_MODE, Optimizer
from HelperFunctions.DataFunctions.DataHelperFunctions import dataloaderSetup
from HelperFunctions.Extras.NameGeneratorFunctions import nameGenerator
from HelperFunctions.General.ModelSetupFunctions import neuralNetSetup
from HelperFunctions.General.SetupHelper import OptimizerAndSchedulerSetup, TrainingModeSetup


device = torch.device('cuda')
confusionmatrixdevice = torch.device('cuda')

# Output Settings
all_classes = ["dog", "flower", "other"]
nothingList = [""]
confusionmatrixeveryXepoch = 0
# Data Settings
DATASETPATH = './datasets/'
all_datasetclasses_train = ["Dog", "Other", "Flower"]
all_transforms_train = ['train']
all_datasetclasses_test = ["Dog", "Other", "Flower"]
all_transforms_test = ['test']
num_workers = 2
pin_memory = True
# Model Settings
MODELFOLDER = './Models/'
#PRETRAINEDFILE = "./RotNetModels/OnlyDogDataLinearLRBS8.pth"
PRETRAINEDFILE = "./JigsawModels/JigsawAcc89Dog.pth"
#PRETRAINEDFILE = "NIX"
param_freeze = False
param_eval = False
pretrained_model = PRETRAINED_MODEL.JIGSAW
# Train Parameter
num_classes = len(all_classes)
num_epochs = 3
batch_size = 3
learning_rate = 0.00001
weight_decay = 0
optimizer_param = Optimizer.ADAM
scheduler_mode = LR_Scheduler.NONE
training_mode = TRAINING_MODE.NORMAL
# SGD optimizer
momentum = 0.9
# ReduceLROnPlateau Scheduler
patience = 100
# CosineAnnealingWarmRestart Scheduler
T_0 = 1


# 1 == True ; 0 == False
save_trained_model = 1

# Automatic Filename for saving
SAVEFILE = nameGenerator(nothingList, num_epochs, batch_size, learning_rate, weight_decay, optimizer_param,
                         scheduler_mode, momentum, MODELFOLDER, T_0, False,
                         customcomment=f"{pretrained_model}")

# Manuel Filename for loading
LOADFILE = "./checkpointsave/Checkpoint_PRETRAINED_MODEL.ROTNETBS60_LR0.0001_W0_NE3_Optimizer.ADAM_LR_Scheduler.NONE_.pth"

# Writer for Tensorboard
Tensorboard_name = nameGenerator(nothingList, num_epochs, batch_size, learning_rate, weight_decay,
                                 optimizer_param, scheduler_mode, momentum, MODELFOLDER, T_0, True,
                                 customcomment=f"{pretrained_model}")

writerPath = f"Tensorboard/runs/TransferLearning/CUDA/{Tensorboard_name}"

#Using Date as a savefile name and using excel file to save the corresponding parameters


def main():
    if os.path.exists(writerPath) and (
            training_mode == TRAINING_MODE.NORMAL \
            or training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TRAINING):
        raise Exception("WARNING! Overwriting FILE! To continue you need to change the customcomment!\n\n")

    writer = SummaryWriter(writerPath)

    model = neuralNetSetup(PRETRAINEDFILE, num_classes, pretrained_model, param_freeze, param_eval).to(device)
    # print(model)
    train_loader, test_loader = dataloaderSetup(DATASETPATH, all_datasetclasses_train, all_transforms_train,
                                                all_datasetclasses_test, all_transforms_test, batch_size,
                                                num_workers, pin_memory)

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
                      scheduler_mode, test_loader, LOADFILE)


if __name__ == "__main__":
    main()
