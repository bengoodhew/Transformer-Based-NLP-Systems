import numpy as np
import torch 
from pathlib import Path

class Logger():
    """
        Handles experiment output management.

        Responsible for:
        - saving training metrics
        - saving and loading model checkpoints
        - storing best model metadata
        - creating experiment directory structure
    """
    def __init__(self, saveFolder):
        self.saveFolder = self.folderInit(saveFolder)
        
    def saveEpoch(self, trainLosses, valLosses, fOnes):
        """
            Save training progress metrics after each epoch.
        """
        self.saveNumpy(trainLosses, "metrics/trainLoss.npy")
        self.saveNumpy(valLosses,  "metrics/valLoss.npy")
        self.saveNumpy(fOnes, "metrics/fOnes.npy")


    def saveModel(self, extension, modelStateDict):
        """
            Save a model checkpoint state dictionary.
        """
        torch.save(modelStateDict, self.saveFolder / "models" / extension)

    
    def loadModel(self, extension):
        """
            Load a previously saved model checkpoint.
        """
        data = torch.load(self.saveFolder / "models" / extension)
        return data

    def saveBestInfo(self, bestFone, bestEpoch, threshold = None):
        """
            Store metadata about the best performing checkpoint.

            Threshold is optional because it is only selected for the sentiment evaluation threshold.
        """
        toWrite = f"The best models validation fOne was:{bestFone} and it was achived at epoch:{bestEpoch}"

        if threshold:
            toWrite = toWrite + f" a threshold of {threshold:.2f} was used"

        with open(self.saveFolder / "models/bestInfo.txt", "w") as fp:
            fp.write(toWrite)

    def saveNumpy(self, data, fileName):
        """
            Save experiment metrics as NumPy arrays.
        """
        with open(self.saveFolder / fileName, "wb") as fp:
            np.save(fp, np.array(data))

    def folderInit(self, saveFolder):
        """
        Creates the experiment directory structure.

        Example:
        experiments/
            taskName/
                metrics/
                models/
        """
        root = Path(__file__).resolve().parents[2]
        baseFolder = root / "experiments" / saveFolder
        metricsFolder = baseFolder / "metrics"
        modelFolder = baseFolder / "models"
        metricsFolder.mkdir(parents=True, exist_ok=True)
        modelFolder.mkdir(exist_ok=True)
        return baseFolder

