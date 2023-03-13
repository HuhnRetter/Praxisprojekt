#General

This package is split into four separate files, which are grouped by the following topics:

**ModelSetupFunctions** <br />
This file contains functions for loading and initializing models
- class JigsawNetwork
- rotNetSetup
- neuralNetSetup
- load_model

**PhaseHelperFunctions** <br />
This file contains helper functions for the training and testing phase.
- drawTensorboardGraph
- createCheckpointsave
- createGraph
- showExamples
- convertFloatTensorToLongTensor

**Phases** <br />
This file contains the main function used for training and testing models
- trainingPhase
- testingPhase
- validationPhase
- estimateLearningRateForCLR

**SetupHelper** <br />
This file contains functions for handling the different parameters
- OptimizerAndSchedulerSetup
- TrainingModeSetup
