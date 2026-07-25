import torch
import numpy as np

import time


def trainModel(trainData, valData, model, optimiser, criterion, evaluator, epochs, log, device):
    """
        Shared training loop for all NLP tasks.

        Each task provides its own:
        - dataset
        - model
        - loss function
        - evaluator

        while using the same optimisation and validation flow.

        The evaluator is responsible for handling task specific validation behaviour, still returning f1.
    """
    trainLosses = []
    valLosses = []
    fOnes = []

    bestFone = 0
    bestEpoch = 0

    start = time.time()
    for epoch in range(epochs):
        trainLossStep = []
        model.train()
        for batch in trainData:
            x = batch["input_ids"].to(device)
            attentionMask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optimiser.zero_grad()
            # Mixed precision reduces memory usage while still converting logits back to float32 for numerical stability in loss calcualtion
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids = x, attention_mask = attentionMask, labels = y).logits.float()


            # Convert model outputs and labels into the format expected by CrossEntropyLoss.
            # Token classification tasks (such as NER) contain sequence dimension,
            # which is flattened into individual token predictions -- from [batch, seq, class] to [batch * seq, class]
            # Other tasks function as a do nothing operation staying as [batch, class]
            output = output.view(-1, output.size(-1))
            y = y.view(-1)

            trainLoss = criterion(output, y)

            trainLoss.backward()
            optimiser.step()
            
            trainLossStep.append(trainLoss.item())

        trainLosses.append(np.mean(trainLossStep))


        valStepLosses = []
        model.eval()
        # Validation does not update model parameters, so gradients are disabled to reduce memory usage.
        with torch.no_grad():
            for batch in valData:
                x = batch["input_ids"].to(device)
                attentionMask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(input_ids = x, attention_mask = attentionMask, labels = y).logits.float()

                output = output.view(-1, output.size(-1))
                y = y.view(-1)

                valLoss = criterion(output, y).item()

                # Metrics such as F1 require predictions across the entire validation set, so evaluators 
                # accumulate batch results before calculating the final score.
                evaluator.step(output, y)

                valStepLosses.append(valLoss)

        valFOne = evaluator.eval()


        # Some evaluators return additional information.
        # Currently sentiment returns the classification threshold selected during validation.
        threshold = None
        if isinstance(valFOne, tuple):
            valFOne, threshold = valFOne
            
        
        fOnes.append(valFOne)
        valLosses.append(np.mean(valStepLosses))
        

        log.saveEpoch(trainLosses, valLosses, fOnes)
        log.saveModel("modelLatest.pt", model.state_dict())

        # The best model is selected using validation F1 rather than loss as it better represnts performance on imbalance classes.
        if bestFone < fOnes[-1]:
            bestFone = fOnes[-1]
            bestEpoch = epoch
            log.saveModel("modelBest.pt", model.state_dict())
            log.saveBestInfo(bestFone, bestEpoch, threshold)

        
        print(f"Epoch:{epoch} Got Training Loss:{trainLosses[-1]:.6f} and Validation Loss:{valLosses[-1]:.6f} and Got f1:{fOnes[-1]:.6f} Time:{time.time()-start:.2f}")
        start = time.time()
    print(f"Found the best f1:{bestFone} at the epoch:{bestEpoch}")

