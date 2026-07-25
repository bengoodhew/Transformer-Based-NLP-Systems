import torch
from sklearn.metrics import f1_score, accuracy_score


class evaluator:
    """
        Evaluates anaphora resolution predictions.

        The evaluator accumulates model outputs across batches,
        converts logits into selected candidate indices, and calculates
        macro F1 across the candidate selection classes.

        Candidate labels represent shuffled candidate positions rather than
        semantic categories, macro F1 is used as the main metrics.
    """
    def __init__(self, device):
        self.device = device
        self.epochInit()


    def step(self, pred, label):
        """
            Accumulate predictions and labels from the current batch.
        """
        self.epochLabels = torch.concat((self.epochLabels, label.detach()))
        self.epochPreds = torch.concat((self.epochPreds, pred.detach()))
        

    
    def eval(self):
        """
            Calculate evaluation f1 over all accumulated samples.
        """
        labels = self.epochLabels.cpu().numpy()

        predLabel = torch.argmax(self.epochPreds, dim=-1).cpu().numpy()
        
        
        fOne = f1_score(predLabel, labels, average="macro")
        
        self.epochInit()

        return fOne


    def fullEval(self):
        """
           Perform full evaluation all accumulated samples and return as formatted string for display.w
        """
        labels = self.epochLabels.cpu().numpy()

        predLabel = torch.argmax(self.epochPreds, dim=-1).cpu().numpy()
        
        
        fOne = f1_score(predLabel, labels, average="macro")
        acc = accuracy_score(predLabel, labels)
        
        result = f"Test f1 is:{fOne:.4f} and accuracy is:{acc:.4f}"

        return result
    
    def epochInit(self):
        """
            Reset stored predictions and labels for a new evaluation cycle.
        """
        self.epochPreds = torch.tensor([], device = self.device)
        self.epochLabels = torch.tensor([], device = self.device)

