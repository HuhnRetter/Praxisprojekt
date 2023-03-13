import random

import numpy as np
import torch.optim.lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from HelperFunctions.DataFunctions.CustomDatasets import ImageDataset, GetJigsawPuzzleDataset
from HelperFunctions.DataFunctions.EnumsForParameter import RANGE_INDICES_MODE, DATASET_CHOSEN, RANGE_PERMUT_MODE


def concatMultipleImageDatasets(FULLDATASETPATH, all_datasetclasses, all_transforms):
    """concats multiple ImageDataset classes and transforms them.

    :param FULLDATASETPATH: full path for the used Dataset
    :param all_datasetclasses: list of datasets used (in my case: dog, flower and other)
    :param all_transforms: list of all transformations corresponding to the getDataTransform() function, for Example
    'train'
    :return: returns concat and transformed ImageDataset
    """
    data_transforms = getDataTransforms()
    for i, element in enumerate(all_datasetclasses):
        for j, transform in enumerate(all_transforms):
            if i == 0 and j == 0:
                dataset = ImageDataset(FULLDATASETPATH, element, i, transform=data_transforms[transform])
            else:
                dataset = dataset + ImageDataset(FULLDATASETPATH, element, i, transform=data_transforms[transform])
    return dataset


def concatMultipleJigsawDatasets(FULLDATASETPATH, all_datasetclasses, all_transforms, avail_permuts_file_path,
                                 range_permut_mode, amount_of_permuts=10):
    """concats multiple ImageDataset classes and transforms them.

    :param FULLDATASETPATH: full path for the used Dataset
    :param all_datasetclasses: list of datasets used (in my case: dog, flower and other)
    :param all_transforms: list of all transformations corresponding to the getDataTransform() function, for Example
    'train'
    :param avail_permuts_file_path: full path for the permutation file
    :param range_permut_mode: enum class RANGE_PERMUT_MODE --> "Normal" = pick random Permutation from x to y or "none"
    = pick random Permutation from whole list
    :param amount_of_permuts: integer value for determining how many permutations are used
    :return: returns concat and transformed JigsawDataset
    """
    data_transforms = getDataTransforms()
    max_length = len(np.load(avail_permuts_file_path))
    if not (max_length / amount_of_permuts).is_integer():
        print("Selected amount_of_permuts results in a float. Mode for Permutation selection is changed")
        random_list = random.sample(range(0, max_length), amount_of_permuts)
        dataset = helperJigsawDataset(FULLDATASETPATH, all_datasetclasses, all_transforms, avail_permuts_file_path,
                                      range_permut_mode, random_list, random_list[0], RANGE_INDICES_MODE.FLOAT, 0)

    else:
        use_perm_every_x = int(max_length / amount_of_permuts)
        dataset = helperJigsawDataset(FULLDATASETPATH, all_datasetclasses, all_transforms, avail_permuts_file_path,
                                      range_permut_mode, range(0, max_length, use_perm_every_x), 0,
                                      RANGE_INDICES_MODE.INT, use_perm_every_x)

    return dataset


def dataloaderSetup(DATASETPATH, all_datasetclasses_train, all_transforms_train, all_datasetclasses_test,
                    all_transforms_test, batch_size, num_workers_param=0, pin_memory_param=False,
                    dataset_choosen=DATASET_CHOSEN.IMAGEDATASET, avail_permuts_file_path=None,
                    range_permut_mode=None, amount_of_permuts=10):
    """setups train and test dataloader from the given path
    :param DATASETPATH: folder path for Datasets for example: ./datasets/
    :param all_datasetclasses_train: list of datasets used for training (in my case: dog, flower and other)
    :param all_transforms_train: list of all transformations corresponding to the getDataTransform() function, for Example
    'train'
    :param all_datasetclasses_test: list of datasets used for testing (in my case: dog, flower and other)
    :param all_transforms_test: list of all transformations corresponding to the getDataTransform() function, for Example
    'test'
    :param batch_size: int value for batch size
    :param num_workers_param: same as in DataLoader
    :param pin_memory_param: same as in DataLoader
    :param dataset_choosen: enum class DATASET_CHOSEN for choosing which Custom Dataset is being used
    :param avail_permuts_file_path: full path for the permutation file
    :param range_permut_mode: enum class RANGE_PERMUT_MODE --> "Normal" = pick random Permutation from x to y or "none"
    = pick random Permutation from whole list
    :param amount_of_permuts: integer value for determining how many permutations are used
    :return: returns train and test loader
    """
    Image.LOAD_TRUNCATED_IMAGES = True

    # Image dataset
    if (dataset_choosen == DATASET_CHOSEN.IMAGEDATASET):
        train_dataset = concatMultipleImageDatasets(f'{DATASETPATH}/train', all_datasetclasses_train,
                                                    all_transforms_train)
        test_dataset = concatMultipleImageDatasets(f'{DATASETPATH}/test', all_datasetclasses_test, all_transforms_test)
    if (dataset_choosen == DATASET_CHOSEN.JIGSAWDATASET):
        train_dataset = concatMultipleJigsawDatasets(f'{DATASETPATH}/train', all_datasetclasses_train,
                                                     all_transforms_train, avail_permuts_file_path, range_permut_mode,
                                                     amount_of_permuts)
        test_dataset = concatMultipleJigsawDatasets(f'{DATASETPATH}/test', all_datasetclasses_test,
                                                    all_transforms_test, avail_permuts_file_path, range_permut_mode,
                                                    amount_of_permuts)

    # Data loader
    persistent_worker_param = True
    if (num_workers_param == 0):
        persistent_worker_param = False

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers_param, pin_memory=pin_memory_param,
                                               persistent_workers=persistent_worker_param)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers_param, pin_memory=pin_memory_param,
                                              persistent_workers=persistent_worker_param)
    return train_loader, test_loader


def helperJigsawDataset(FULLDATASETPATH, all_datasetclasses, all_transforms, avail_permuts_file_path,
                        range_permut_mode, second_iterator, comparison, range_permut_indices_mode, use_perm_every_x):
    """used for concatenating JigsawDatasets used as a helper in concatMultipleJigsawDatasets

    :param FULLDATASETPATH: full path for the dataset
    :param all_datasetclasses: list of datasets used (in my case: dog, flower and other)
    :param all_transforms: list of all transformations corresponding to the getDataTransform() function, for Example
    'train'
    :param avail_permuts_file_path: full path for the permutation file
    :param range_permut_mode: enum class RANGE_PERMUT_MODE --> "Normal" = pick random Permutation from x to y or "none"
    = pick random Permutation from whole list
    :param second_iterator: smaller segment of permutation list
    :param comparison: used in checkIfFirstDatasetInConcat for determining if it is the first Dataset
    :param range_permut_indices_mode: enum class RANGE_INDICES_MODE for change between INT and FLOAT values if
    range_permut_mode = normal cant be used
    :param use_perm_every_x: used as an indicator when the next small segment begins (second_iterator)
    :return: returns concat JigsawDataset
    """
    data_transforms = getDataTransforms()
    for i, element in enumerate(all_datasetclasses):
        if range_permut_mode == RANGE_PERMUT_MODE.NORMAL:
            for j in second_iterator:
                if checkIfFirstDatasetInConcat(i, j, comparison):
                    dataset = GetJigsawPuzzleDataset(FULLDATASETPATH, element, avail_permuts_file_path,
                                                     range_permut_indices=get_range_permut_indices(
                                                         j, use_perm_every_x, range_permut_indices_mode),
                                                     transform=data_transforms[all_transforms[0]])
                else:
                    dataset = dataset + GetJigsawPuzzleDataset(FULLDATASETPATH, element, avail_permuts_file_path,
                                                               range_permut_indices=get_range_permut_indices(
                                                                   j, use_perm_every_x, range_permut_indices_mode),
                                                               transform=data_transforms[all_transforms[0]])
        else:
            if i == 0:
                dataset = GetJigsawPuzzleDataset(FULLDATASETPATH, element, avail_permuts_file_path,
                                                 range_permut_indices=None,
                                                 transform=data_transforms[all_transforms[0]])
            else:
                dataset = dataset + GetJigsawPuzzleDataset(FULLDATASETPATH, element, avail_permuts_file_path,
                                                           range_permut_indices=None,
                                                           transform=data_transforms[all_transforms[0]])

    return dataset


def checkIfFirstDatasetInConcat(i, j, comparison):
    """helper function for checking if it is the first Dataset used for concat

    :param i: counter for all_datasetclasses
    :param j: counter for second_iterator
    :param comparison: first element from random sample
    :return: returns if it is the first Dataset used for concat
    """
    return i == 0 and j == comparison


def get_range_permut_indices(j, use_perm_every_x, range_permut_indices_mode):
    """helper function for determining interval of small segment

    :param j: counter for second_iterator
    :param use_perm_every_x: used as an indicator when the next small segment begins (second_iterator)
    :param range_permut_indices_mode: enum class RANGE_INDICES_MODE for change between INT and FLOAT values if
    range_permut_mode = normal cant be used
    :return: returns interval used for random pick permutation
    """
    if (range_permut_indices_mode == RANGE_INDICES_MODE.FLOAT):
        return [j, j]
    else:
        return [j, j + use_perm_every_x - 1]


def getDataTransforms():
    """contains all Data Transforms

    :return: returns all Data Transforms
    """
    # every rot +90 degrees because the images are rotated by 90 degrees
    data_transforms = {
        'rot0': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'rot90': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'rot180': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(270, 270)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'rot270': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(0, 0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'jigsaw': transforms.Compose([
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),

    }
    return data_transforms

