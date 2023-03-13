from HelperFunctions.DataFunctions.EnumsForParameter import LR_Scheduler, Optimizer


def nameGenerator(all_datasetclasses_train, num_epochs, batch_size, learning_rate, weight_decay, optimizer_param,
                  scheduler_mode, momentum, MODELFOLDER, T_0, forTensorboard, customcomment=""):
    """Creating Filename with given parameters

    :param all_datasetclasses_train: list of datasets used for training (in my case: dog, flower and other)
    :param num_epochs: max_number of epoch
    :param batch_size: batch size used
    :param learning_rate: learning rate used
    :param weight_decay: weight decay used
    :param optimizer_param: enum class Optimizer for choosing opitimier
    :param scheduler_mode: enum class Scheduler for choosing scheduler
    :param momentum: momentum parameter for SGD optimizer
    :param MODELFOLDER: folder path for saving trained model
    :param T_0: T_0 parameter for COSINEANNEALINGWARMRESTARTS scheduler
    :param forTensorboard: boolean value for choosing if nameGenerator is used for tensorboard
    :param customcomment: adds a custom comment at the end of the name
    :return: returns string containing the filename/filepath
    """
    # batch_size - learning_rate - weight_decay - num_epochs - optimizer_param - optimizer_specific settings - scheduler_mode
    customChar = "."
    MiddleFilename = f"BS{batch_size}_LR{changeDotToCustom(learning_rate, customChar)}_W{changeDotToCustom(weight_decay, customChar)}_NE{num_epochs}_{optimizer_param}_"
    if (optimizer_param == Optimizer.SGD):
        MiddleFilename = MiddleFilename + f"M{changeDotToCustom(momentum, customChar)}_"

    MiddleFilename = MiddleFilename + f"{scheduler_mode}_"
    if (scheduler_mode == LR_Scheduler.COSINEANNEALINGWARMRESTARTS):
        MiddleFilename = MiddleFilename + f"T{T_0}_"

    # Training data usage
    for element in all_datasetclasses_train:
        MiddleFilename = MiddleFilename + f"{element}"

    if (forTensorboard == True):
        return MiddleFilename + f"{customcomment}"

    EndFilename = ".pth"
    return f"{MODELFOLDER}{customcomment}{MiddleFilename}{EndFilename}"


def changeDotToCustom(floatVariable, customChar):
    """changes Dot character to a choosen character

    :param floatVariable: float used for changing the dot character
    :param customChar: custom character used for replacing the dot
    :return: returns string with replaced dot character
    """
    return str(floatVariable).replace('.', customChar)
