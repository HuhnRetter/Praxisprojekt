import math

import torch
from matplotlib import pyplot as plt

from HelperFunctions.Extras.ConfusionMatrixFunctions import createConfusionmatrixEveryXepoch


def drawTensorboardGraph(writer, running_loss, running_correct, n_total_steps, n_total_steps_quarter, epoch, num_epochs,
                         i,
                         predicted, verbose=True):
    """draws graph of accuracy and training loss in tensorboard

    :param writer: writer for the Tensorboard
    :param running_loss: sum of all loses from the learning
    :param running_correct: sum of all correct guesses
    :param n_total_steps: total steps/batches of the epoch as int
    :param n_total_steps_quarter: variable to determine after every x steps add scalar to writer
    :param epoch: current epoch as int
    :param num_epochs: total epochs to run
    :param i: current step as int
    :param predicted: the predicted class after input into the model
    :param verbose: boolean value to determine if an output is displayed
    """
    loss = running_loss / n_total_steps_quarter
    # to fix faulty outputs
    if loss > 2:
        print("\n[Info] loss higher than 2 -> reduced to fix output bugs\n")
        loss = 2
    if math.isnan(loss):
        print("\n[Info] loss is nan -> reduced to fix output bugs\n")
        loss = 2

    writer.add_scalar('training loss', loss, epoch * n_total_steps + i)
    running_accuracy = running_correct / n_total_steps_quarter / predicted.size(0)
    if (verbose == True):
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Accuracy: {running_accuracy:.4f}, Loss: {loss:.4f}\n\r')
    writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
    writer.close()


def createCheckpointsave(model, epoch, device, confusionmatrixeveryXepoch, train_loader, all_classes,
                         confusionmatrixdevice, writer, FILE, tensorboardConfusionMatrix=True):
    """creates a checkpoint save for the given model

    :param model: current model of the class NeuralNet
    :param epoch: current epoch
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param confusionmatrixeveryXepoch: value for every x epoch create confusion matrix if 0 then only create confusion matrix at the end
    :param train_loader: dataloader with Training dataset
    :param all_classes: list of all possible classes
    :param confusionmatrixdevice: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String (used in tensorboard)
    :param tensorboardConfusionMatrix: boolean value for determining if confusionmatrix should be created
    """
    model.to(torch.device("cpu"))
    splitted_string = FILE.split("/")
    MiddleFilename = splitted_string[2]
    # EndFilename = ".pth"
    CHECKPOINTFILE = f"./checkpointsave/Checkpoint_{MiddleFilename}"
    torch.save(model.state_dict(), CHECKPOINTFILE)
    model.to(torch.device(device))
    if tensorboardConfusionMatrix == True:
        createConfusionmatrixEveryXepoch(model, epoch, confusionmatrixeveryXepoch, train_loader, all_classes,
                                         confusionmatrixdevice, writer, FILE)


def createGraph(learning_rates_x, accuracy_y):
    """creates a graph by using a list for learning rates and one for the corresponding accuracy

    :param learning_rates_x: list of learning rates
    :param accuracy_y: list of accuracy
    """
    fig, ax = plt.subplots()
    ax.plot(learning_rates_x, accuracy_y)
    plt.show()


def showExamples(train_loader, all_classes, columns=4):
    """shows examples from the given dataloader

    :param train_loader: dataloader with Training dataset
    :param all_classes: list of all possible classes
    """
    import numpy as np
    # columns = 4
    # rows = 3
    # fig, ax = plt.subplots(rows, columns, dpi=224)
    # ax = ax.ravel()
    # for i, (image, labels) in enumerate(train_loader):
    #     if i == rows:
    #         break
    #     for j, (img) in enumerate(image):
    #         np_array = img.numpy()
    #         np_array = np_array.swapaxes(0, 2)
    #         # np_array = np.rot90(np_array, 3)
    #         ax[i * columns + j].imshow(np_array)
    #         ax[i * columns + j].set_title(all_classes[labels[j]])  # set title
    # fig.tight_layout()
    # plt.show()
    fig, ax = plt.subplots(len(all_classes), columns, dpi=224)
    ax = ax.ravel()
    for i,c in enumerate(all_classes):
        for j in range(columns):
            currentlabel = -1
            while currentlabel != i:
                image, labels = next(iter(train_loader))
                for (clabel, cimage) in zip(labels,image):
                    if clabel.item() == i:
                        currentlabel = clabel
                        currentimage = cimage
                        break

            np_array = currentimage.numpy()
            np_array = np_array.swapaxes(0, 2)
            np_array = np.rot90(np_array, 3)
            ax[i * columns + j].imshow(np_array)
            ax[i * columns + j].set_title(c)  # set title
    fig.tight_layout()
    plt.show()



def convertFloatTensorToLongTensor(floatTensor):
    """converts a float tensor to a long tensor

    :param floatTensor: tensor of type float
    :return: returns a tensor of type long
    """
    # Axis correction
    floatTensor = floatTensor.view(floatTensor.size(dim=0))
    # Convert to LongTensor
    longTensor = floatTensor.long()
    return longTensor
