import time
import torch
from matplotlib import pyplot as plt

from HelperFunctions.DataFunctions.EnumsForParameter import LR_Scheduler, TEST_MODE
from HelperFunctions.Extras.ConfusionMatrixFunctions import createConfusionmatrixEveryXepoch, createConfusionMatrix
from HelperFunctions.Extras.ProgressbarFunctions import progressBarWithTime
from HelperFunctions.General.PhaseHelperFunctions import convertFloatTensorToLongTensor, drawTensorboardGraph, createCheckpointsave, \
    createGraph


def trainingPhase(model, criterion, optimizer, scheduler, train_loader, num_epochs, print_every_x_percent, save, device,
                  confusionmatrixdevice, writer, FILE, all_classes, label_start_at, confusionmatrixeveryXepoch,
                  scheduler_mode, test_loader, use_val_Phase):
    """trains the given model with the given parameters.

    iterates once through the train_loader in each epoch
    and updates the weights in the model

    After every x step update the acc and loss graph in Tensorboard
    and after every epoch create Confusion matrix

    :param model: current model of the class NeuralNet
    :param criterion: loss function from torch.nn.modules.loss (for Example CrossEntropyLoss)
    :param optimizer: optimizer from torch.optim (for Example Adam)
    :param scheduler: scheduler from torch.optim.scheduler
    :param train_loader: dataloader with Training dataset
    :param num_epochs: number of training epochs
    :param print_every_x_percent: used to determine when Tensorboard is updated
    :param save: parameter to determine if the model should be saved --> 1 == true & 0 == false
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param confusionmatrixdevice: device on which the tensors are being processed for creating the confusionmatrix example: "cuda", "cpu", "dml"
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param confusionmatrixeveryXepoch: value for every x epoch create confusion matrix if 0 then only create confusion matrix at the end
    :param scheduler_mode: enum class LR_Scheduler for determining which scheduler is being used
    :param test_loader: dataloader with testing dataset for the val phase
    :param use_val_Phase: boolean value for choosing to use a val phase
    :return: returns trained model of the class NeuralNet
    """
    n_total_steps = len(train_loader)
    print(f"n_total_steps: {n_total_steps}\n\n")
    n_every_steps = int(n_total_steps * print_every_x_percent)
    print(f"n_every_steps: {n_every_steps}\n\n")
    running_loss = 0.0
    running_correct = 0
    sum_loss = 0
    starting_time = time.time()
    highest_acc = 0
    model.to(device)
    for epoch in range(num_epochs):
        if epoch + 1 > 1:
            string_output = "\nTraining Resumed\n"
            print(f"{string_output:#^{(len(string_output) - 2) * 3}}")

        for i, (hsl, labels) in enumerate(train_loader):
            if label_start_at == 1:
                # Transform labels if start with 1
                labels = torch.add(labels, -1)
            # Forward pass
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels).to(device)
            # print(f"labels: {labels}\n\n")
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_mode == LR_Scheduler.COSINEANNEALINGWARMRESTARTS:
                scheduler.step((epoch + i) / n_total_steps)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # print(f"predicted: {predicted}\n\n")
            running_correct += (predicted == labels).sum().item()
            if ((i + 1) % n_every_steps == 0):  # or ((i + 1) == n_total_steps): Produces wrong calculations
                ##############SCHEDULER REDUCEONPLATEAU############
                sum_loss += running_loss
                curr_acc = running_correct / n_every_steps / predicted.size(0)
                ############## TENSORBOARD ########################
                drawTensorboardGraph(writer, running_loss, running_correct, n_total_steps, n_every_steps, epoch, num_epochs, i,
                                     predicted)
                running_correct = 0
                running_loss = 0.0
                ###################################################
                # Checkpointsave
                if curr_acc > highest_acc:
                    string_output = "\nHighest_Avg_Acc saved as a Checkpoint\n"
                    print(f"{string_output:#^{(len(string_output) - 2) * 3}}")
                    createCheckpointsave(model, epoch, device, confusionmatrixeveryXepoch, train_loader, all_classes,
                                         confusionmatrixdevice, writer, FILE, tensorboardConfusionMatrix=False)
                    highest_acc = curr_acc

            progressBarWithTime(i + n_total_steps * epoch, n_total_steps * num_epochs, starting_time)
        # Save confusion matrix to Tensorboard
        if epoch + 1 == num_epochs:
            string_output = "\nTraining Completed\n"
            print(f"{string_output:#^{(len(string_output) - 2) * 3}}")
        else:
            string_output = "\nTraining Paused\n"
            print(f"{string_output:#^{(len(string_output) - 2) * 3}}")
        avg_loss = sum_loss / n_total_steps
        if scheduler_mode == LR_Scheduler.REDUCELR0NPLATEAU:
            scheduler.step(avg_loss)
        # Create Confusionmatrix
        if epoch + 1 != num_epochs:
            createConfusionmatrixEveryXepoch(model, epoch, confusionmatrixeveryXepoch, train_loader, all_classes,
                                             confusionmatrixdevice, writer, FILE)
        if use_val_Phase:
            validationPhase(model, device, test_loader, epoch, n_total_steps, writer, True)

    model.to(torch.device("cpu"))
    if save == 1:
        torch.save(model.state_dict(), FILE)
    createConfusionmatrixEveryXepoch(model, num_epochs, 1, train_loader, all_classes,
                                     confusionmatrixdevice, writer, FILE)
    return model


def testingPhase(model, test_loader, writer, FILE, all_classes, label_start_at, device,
                 test_mode=TEST_MODE.CONFUSION_MATRIX):
    """tests the model
    creates confusion matrix or outputs the average accuracy


    :param model: current model of the class NeuralNet
    :param test_loader: dataloader with Test dataset
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param test_mode: enum class TEST_MODE for choosing to use a confusion matrix or an average accuracy
    """
    with torch.no_grad():
        string_output = "\nStarting with Testing!\n"
        print(f"{string_output:#^{(len(string_output) - 2) * 3}}")
        # Save confusion matrix to Tensorboard
        print(len(test_loader))
        if (test_mode == TEST_MODE.CONFUSION_MATRIX):
            currentConfusionMatrix = createConfusionMatrix(test_loader, model, all_classes, label_start_at,
                                                           device)
            plt.show()
            writer.add_figure(f"Confusion matrix testing from: {FILE}", currentConfusionMatrix)
            writer.close()
        else:
            validationPhase(model, device, test_loader)


def validationPhase(model, device, test_loader, epoch=0, n_total_steps_train_loader=0, writer=None,
                    tensorboard_verbose=False):
    """determines the accuracy of correct classification from the given test loader and model

    :param model: current model of the class NeuralNet
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param test_loader: dataloader with Test dataset
    :param epoch: current epoch from the function trainingPhase
    :param n_total_steps_train_loader: n_total_steps_train_loader from the function trainingPhase
    :param writer: writer for the Tensorboard
    :param tensorboard_verbose: boolean value to choose if the result should be used in tensorboard
    """
    with torch.no_grad():
        model.to(device)
        n_total_steps = len(test_loader)
        running_correct = 0
        starting_time = time.time()
        if tensorboard_verbose:
            string_output = "\nStarting with Val_Phase!\n"
            print(f"{string_output:#^{(len(string_output) - 2) * 3}}")

        for i, (hsl, labels) in enumerate(test_loader):
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels).to(device)

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            progressBarWithTime(i, n_total_steps, starting_time)

        running_accuracy = running_correct / n_total_steps / predicted.size(0)
        if tensorboard_verbose == True:
            if epoch == 0:
                writer.add_scalar('val_value', 0, 0)
            writer.add_scalar('val_value', running_accuracy, (epoch+1) * n_total_steps_train_loader)
            print(f'Epoch [{epoch + 1}], Val_value: {running_accuracy:.4f}\n\r')
        else:
            print(f'Avg_accuracy: {running_accuracy:.4f}\n\r')


def estimateLearningRateForCLR(model, criterion, optimizer, scheduler, train_loader, num_epochs,
                               device, label_start_at, all_classes):
    """trains a model with a linear increasing learning rate

    :param model: current model of the class NeuralNet
    :param criterion: loss function from torch.nn.modules.loss (for Example CrossEntropyLoss)
    :param optimizer: optimizer from torch.optim (for Example Adam)
    :param scheduler: scheduler from torch.optim.scheduler
    :param train_loader: dataloader with Training dataset
    :param num_epochs: number of training epochs
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param all_classes: list of all possible classes
    :return: returns trained model of the class NeuralNet
    """
    n_total_steps = len(train_loader)
    running_correct = 0
    starting_time = time.time()
    model.to(device)
    learning_rates_x = []
    accuracy_y = []
    print("\n############################Estimate Learning Rate Started#################################\n")
    for epoch in range(num_epochs):
        if epoch + 1 > 1:
            print("\n############################Estimate Learning Rate Resumed#################################\n")

        for i, (hsl, labels) in enumerate(train_loader):
            if label_start_at == 1:
                # Transform labels if start with 1
                labels = torch.add(labels, -1)

            # Forward pass
            outputs = model(hsl.to(device))
            labels = convertFloatTensorToLongTensor(labels).to(device)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            accuracy = running_correct / (i + 1) / predicted.size(0)
            schedulerlearningRate = float(scheduler.get_last_lr()[0])
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Accuracy: {accuracy:.4f}, LearningRate: {schedulerlearningRate:.4f}                                                \n\r')
            learning_rates_x.append(schedulerlearningRate)
            accuracy_y.append(accuracy)
            progressBarWithTime(i + n_total_steps * epoch, n_total_steps * num_epochs, starting_time)
        # Save confusion matrix to Tensorboard
        if epoch + 1 == num_epochs:
            print("\n############################Estimate Learning Rate Completed###############################\n")
        else:
            print("\n############################Estimate Learning Rate Paused##################################\n")
        createGraph(learning_rates_x, accuracy_y)
        currentConfusionMatrix = createConfusionMatrix(train_loader, model, all_classes, label_start_at, device)
        plt.show()
        running_correct = 0
        # Checkpointsave
        createCheckpointsave(model, epoch, device, None, train_loader, all_classes,
                             None, None, None, False)
    model.to(torch.device("cpu"))
    torch.save(model.state_dict(), "DogPerfectDataLinearLRBS8.pth")
    currentConfusionMatrix = createConfusionMatrix(train_loader, model, all_classes, label_start_at, device)
    plt.show()
    return model
