import time

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from HelperFunctions.Extras.ProgressbarFunctions import progressBarWithTime



def createConfusionMatrix(loader, model, all_classes, label_start_at, device):
    """creates Confusionmatrix from given Dataloader and given Model & gives out a console output

    :param loader: An instance of the class ColorDataset
    :param model: current model of the class NeuralNet
    :param all_classes: list of all possible classes
    :param label_start_at: value for correction of all labels in the dataset if starts at 1
    :param device: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :return: returns confusion matrix as figure
    """
    from HelperFunctions.General.PhaseHelperFunctions import convertFloatTensorToLongTensor


    y_pred = []  # save prediction
    y_true = []  # save ground truth
    # iterate over data

    starting_time = time.time()

    print("\nCreating Confusion Matrix ...")

    n_total_steps = len(loader)

    model.to(device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            output = model(inputs.to(device))  # Feed Network

            if label_start_at == 1:
                # Label transforms because labels start with 1
                labels = torch.add(labels, -1)

            # for output in console
            longLabel = convertFloatTensorToLongTensor(labels)
            _, predicted = torch.max(output.data, 1)

            output = (torch.max(torch.exp(output), 1)[1]).cpu().numpy()
            y_pred.extend(output)  # save prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth
            # print to see progress because sometimes it takes a while and pauses
            progressBarWithTime(i, n_total_steps, starting_time)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * len(all_classes), index=[i for i in all_classes],
                         columns=[i for i in all_classes])

    plt.figure(figsize=(21, 7))
    return sn.heatmap(df_cm.round(4), annot=True).get_figure()

def createConfusionmatrixEveryXepoch(model, epoch, confusionmatrixeveryXepoch, train_loader, all_classes,
                                     confusionmatrixdevice, writer, FILE):
    """creates a confusion matrix every X epoch

    :param model: current model of the class NeuralNet
    :param epoch: current epoch
    :param confusionmatrixeveryXepoch: value for every x epoch create confusion matrix if 0 then only create confusion matrix at the end
    :param train_loader: dataloader with Training dataset
    :param all_classes: list of all possible classes
    :param confusionmatrixdevice: device on which the tensors are being processed for example: "cuda", "cpu", "dml"
    :param writer: writer for the Tensorboard
    :param FILE: path of the model to be loaded as a String (used in tensorboard)
    """
    if confusionmatrixeveryXepoch == 0:
        print("Confusion Matrix will not be created\n\n")
        return

    if (epoch + 1) % confusionmatrixeveryXepoch == 0:
        currentConfusionMatrix = createConfusionMatrix(train_loader, model, all_classes, 0,
                                                       confusionmatrixdevice)
        plt.show()
        writer.add_figure(f"Confusion matrix training from: {FILE}", currentConfusionMatrix, epoch)
        writer.close()
