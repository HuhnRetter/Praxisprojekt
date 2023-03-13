# DataFunctions

This package is split into three separate files, which are grouped by the following topics:

**CustomDatasets** <br />
This file contains custom Datasets, which are subclasses from torch.utils.data.Dataset
- ImageDataset
  - Dataset used for Training the Rotation Pretext Task and Downstream Task
- GetJigsawPuzzleDataset
  - Dataset used for Training the Jigsaw Puzzle Pretext Task

**DataHelperFunctions** <br />
This file contains the functions need for creating and transforming the previously mentioned Datasets. 
Here is a list of all the functions:
- concatMultipleImageDatasets
- concatMultipleJigsawDatasets
- dataloaderSetup
- helperJigsawDataset
- checkIfFirstDatasetInConcat
- get_range_permut_indices
- getDataTransforms
- get_nine_crops
- randomCrop

**EnumsForParameter** <br />
This file contains all used Enum Classes in the project
