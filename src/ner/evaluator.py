import torch
from sklearn.metrics import f1_score, confusion_matrix


class evaluator:
    """
        Evaluates named entity recognition predictions.

        The evaluator accumulates model outputs across batches,
        converts logits into predicted label indices, and calculates
        macro F1 across the selected entities.

        labels correspond to entity class indices. Macro F1 is used as the main metrics.
    """
    def __init__(self, device, idToLabel):
        self.device = device
        self.idToLabel = idToLabel # Only used in test -- fullEval
        self.epochInit()


    def step(self, pred, label):
        """
            Accumulate predictions and labels from the current batch.
            Converts the predictions into label to save vram.
        """
        self.epochLabels = torch.concat((self.epochLabels, label.detach()))

        predLabel = torch.argmax(pred.detach(), dim=-1) # Prediction immediately converted to label indices to reduce memory usage
        self.epochPreds = torch.concat((self.epochPreds, predLabel))

    
    def eval(self):
        """
            Calculate evaluation f1 over all accumulated samples.
            First masked the non-entity marked with -100 to ignore them,
            ensure metrics calculated over only valid entity labels.
        """
        mask  = self.epochLabels != -100
        maskedlabels = self.epochLabels[mask].cpu().numpy()
        maksedPreds = self.epochPreds[mask].cpu().numpy()
        
        
        fOne = f1_score(maksedPreds, maskedlabels, average="macro")
        
        self.epochInit()

        return fOne


    def fullEval(self):
        """
           Perform full evaluation all accumulated samples and return as formatted string for display.

           Calculates the full entity level F1 per class
        """

        mask  = self.epochLabels != -100
        maskedlabels = self.epochLabels[mask].cpu().numpy()
        maksedPreds = self.epochPreds[mask].cpu().numpy()

        
        
        macroFone = f1_score(maksedPreds, maskedlabels, average="macro")
        weightedFone = f1_score(maksedPreds, maskedlabels, average="weighted")

        result = f"Test macro F1 is:{macroFone:.4f} and the weighted F1 is:{weightedFone:.4f}\nThe full class F1's are:"

        classFone = f1_score(maksedPreds, maskedlabels, average=None)
        for i, score in enumerate(classFone):
            result = result + f"\n{self.idToLabel[i]} -- {score:.4f}"

        

        return result


    def epochInit(self):
        """
            Reset stored predictions and labels for a new evaluation cycle.
        """
        self.epochPreds = torch.tensor([], device = self.device)
        self.epochLabels = torch.tensor([], device = self.device)

        

