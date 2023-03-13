import torch.optim.lr_scheduler

from HelperFunctions.DataFunctions.EnumsForParameter import LR_Scheduler, Optimizer, TRAINING_MODE, TEST_MODE
from HelperFunctions.General.ModelSetupFunctions import load_model
from HelperFunctions.General.Phases import estimateLearningRateForCLR, testingPhase, trainingPhase


def OptimizerAndSchedulerSetup(optimizer_param, scheduler_mode, model, learning_rate, weight_decay, momentum, T_0,
                               patience):
    """used for choosing an optimizer and scheduler

    :param optimizer_param: enum class Optimizer used for choosing an optimizer
    :param scheduler_mode: enum class LR_Scheduler for determining which scheduler is being used
    :param model: current model of the class NeuralNet
    :param learning_rate: learning rate used for training the model
    :param weight_decay: param of Optimizer
    :param momentum: param of SGD Optimizer
    :param T_0: param of CosineAnnealingWarmRestarts Scheduler
    :param patience: param of ReduceLROnPlateau Scheduler
    :return: optimizer and scheduler
    """
    if optimizer_param == Optimizer.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_param == Optimizer.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    if scheduler_mode == LR_Scheduler.REDUCELR0NPLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)
    elif scheduler_mode == LR_Scheduler.COSINEANNEALINGWARMRESTARTS:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler

    return optimizer, scheduler


def TrainingModeSetup(training_mode, model, criterion, optimizer, scheduler, train_loader, num_epochs,
                      save_trained_model,
                      device,
                      confusionmatrixdevice, writer, SAVEFILE, all_classes, confusionmatrixeveryXepoch,
                      scheduler_mode, test_loader, LOADFILE=0, base_lr=0, max_lr=0, dataset_size=0, batch_size=0,
                      cycle_epochs=0, test_mode=TEST_MODE.CONFUSION_MATRIX, use_val_Phase=True):
    """

    :param training_mode: enum class TRAINING_MODE for choosing different phases
    :param model: current model of the class NeuralNet
    :param criterion: loss function from torch.nn.modules.loss (for Example CrossEntropyLoss)
    :param optimizer: optimizer from torch.optim (for Example Adam)
    :param scheduler: scheduler from torch.optim.scheduler
    :param train_loader: dataloader with Training dataset
    :param num_epochs: number of training epochs
    :param save_trained_model: parameter to determine if the model should be saved --> 1 == true & 0 == false
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param confusionmatrixdevice: device on which the tensors are being processed for creating the confusionmatrix example: "cuda", "cpu", "dml"
    :param writer: writer for the Tensorboard
    :param SAVEFILE: path of the model to be saved as a String
    :param all_classes: list of all possible classes
    :param confusionmatrixeveryXepoch: value for every x epoch create confusion matrix if 0 then only create confusion matrix at the end
    :param scheduler_mode: enum class LR_Scheduler for determining which scheduler is being used
    :param test_loader: dataloader with testing dataset
    :param LOADFILE: path of the model to be loaded from as a String
    :param base_lr: param of CyclicLR Scheduler
    :param max_lr: param of CyclicLR Scheduler
    :param dataset_size: size of the used dataset (only used as a param for CyclicLR Scheduler)
    :param batch_size: batch size (only used as a param for CyclicLR Scheduler)
    :param cycle_epochs: param of CyclicLR Scheduler
    :param test_mode: enum class TEST_MODE for choosing to use a confusion matrix or an average accuracy
    :param use_val_Phase: boolean value for choosing to use a val phase
    """
    if training_mode == TRAINING_MODE.ESTIMATE_LR:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                                                      step_size_up=int((dataset_size / batch_size) * cycle_epochs))
        model = estimateLearningRateForCLR(model, criterion, optimizer, scheduler, train_loader, num_epochs,
                                           device, 0, all_classes)
        return

    if training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TESTING:
        model = load_model(model, LOADFILE, training_mode)
        testingPhase(model, test_loader, writer, LOADFILE, all_classes, 0, confusionmatrixdevice, test_mode)
        return

    if training_mode == TRAINING_MODE.LOAD_EXISTING_MODEL_TRAINING:
        model = load_model(model, LOADFILE, training_mode)
        model = trainingPhase(model, criterion, optimizer, scheduler, train_loader, num_epochs, 0.1,
                              save_trained_model,
                              device,
                              confusionmatrixdevice, writer, LOADFILE, all_classes, 0, confusionmatrixeveryXepoch,
                              scheduler_mode, test_loader, use_val_Phase)
        testingPhase(model, test_loader, writer, LOADFILE, all_classes, 0, confusionmatrixdevice)
        return

    if training_mode == TRAINING_MODE.NORMAL:
        model = trainingPhase(model, criterion, optimizer, scheduler, train_loader, num_epochs, 0.1,
                              save_trained_model,
                              device,
                              confusionmatrixdevice, writer, SAVEFILE, all_classes, 0, confusionmatrixeveryXepoch,
                              scheduler_mode, test_loader, use_val_Phase)
        testingPhase(model, test_loader, writer, SAVEFILE, all_classes, 0, confusionmatrixdevice)
        return
