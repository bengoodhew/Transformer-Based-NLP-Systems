import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

class evaluator:
    def __init__(self, fitMode = True, device = "cpu", threshold = None):
        """
            Evaluates sentiment analysis predictions.

            The evaluator accumulates model outputs across batches.
            
            During training the evaluator searches over multiple decision
            thresholds to maximise validation macro F1.

            The best threshold is later reused during inference to improve
            performance on the imbalanced dataset.
        """
        self.fitMode = fitMode

        # During inference the threshold has already been found on the validation set and should be used.
        if not self.fitMode: 
            if threshold is None:
                raise TypeError("Must give a threshold if in inference mode")
            self.threshold = threshold

        self.device = device
        
        self.thresholds = np.arange(0.2, 0.9, 0.01) # Candidate thresholds evaluated during validation.

        self.epochInit()
        

    def step(self, pred, label):
        """
            Accumulate predictions and labels from the current batch.
        """
        self.epochPreds = torch.concat((self.epochPreds, pred.detach()))
        self.epochLabels = torch.concat((self.epochLabels, label.detach()))
    
    def eval(self):
        """
            Calculate evaluation f1 over all accumulated samples.
            First softmax to standardise probabilities then apply threshold and calculate f1.
        """
        probs = torch.softmax(self.epochPreds, dim=-1)
        labels = self.epochLabels.cpu().numpy()
        
        for thresh in self.thresholds:
            fOne = self.findFone(probs, labels, thresh)

            if fOne > self.bestVal:
                self.bestThrehold = thresh
                self.bestVal = fOne
                
        bestFound = self.bestVal, self.bestThrehold
        
        self.epochInit()

        return bestFound


    def fullEval(self):
        """
           Perform full evaluation all accumulated samples and return as formatted string for display.

           Calculates the full entity level F1 per class
        """
        probs = torch.softmax(self.epochPreds, dim=-1)
        labels = self.epochLabels.cpu().numpy()

        fOne = self.findFone(probs, labels, self.threshold)
        fullFone = self.findFone(probs, labels, self.threshold, method=None)

        confusionMatrix = self.findConfsuionMatix(probs, labels, self.threshold)

        result = f"Test macro F1 is:{fOne:.4f}\nThe non-offensive F1 is:{fullFone[0]:.4f} the offenive F1 is:{fullFone[1]:.4f}\nThe confusion matix is:\n{confusionMatrix}"
        

        return result



    def findFone(self, probs, labels, threshold, method = "macro"):
        """
            Converts logits into class probabilities before evaluation.

            During training every candidate threshold is evaluated and the
            threshold producing the highest validation F1 is returned.

            The predicted class is taken from the thresholded output rather than
            the maximum probability alone, allowing the decision boundary to be tuned.

            During inference the previously selected threshold is reused.
        """
        
        thredProbs = (probs >= threshold).int() # Apply the current decision threshold to both output probabilities.
        preds = torch.argmax(thredProbs, dim = -1).cpu().numpy()

        fOne = f1_score(preds, labels, average=method)

        return fOne
    
    def epochInit(self):
        """
            Reset stored predictions and labels for a new evaluation cycle.
        """
        self.epochPreds = torch.tensor([], device = self.device)
        self.epochLabels = torch.tensor([], device = self.device)

        self.bestThrehold = None
        self.bestVal = -1

    def findConfsuionMatix(self, probs, labels, threshold):
        thredProbs = (probs >= threshold).int()
        preds = torch.argmax(thredProbs, dim = -1).cpu().numpy()

        return confusion_matrix(preds, labels)
        

