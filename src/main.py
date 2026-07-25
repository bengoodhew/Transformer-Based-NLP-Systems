import numpy as np
import torch
import random
import sys

# Hugging Face transformer models used for the different task types
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForMultipleChoice

# Task-specific data processing modules
from anaphora import data as anaphoraData
from ner import data as nerData
from sentiment import data as senitmentData

# Task-specific evaluation implementations
from sentiment.evaluator import evaluator as sentEval
from ner.evaluator import evaluator as nerEval
from anaphora.evaluator import evaluator as anaEval

# Shared experiment utilities
from common.logger import Logger
from common import trainLoop
from common import testLoop

"""
Entry point for the NLP multi-task experiments.

This project implements three NLP tasks:
- Sentiment classification
- Named entity recognition
- Anaphora resolution

Each task contains task-specific:
- data preprocessing
- evaluation logic
- model configuration

while sharing the same training and testing workflow.
This allows different NLP problem types to be evaluated
through a consistent experiment interface.

Usage:
    python main.py ner train
    python main.py sentiment test
"""



def main():
    numPro = 12  # Number of parallel processes used during dataset preprocessing

    try:
        modelType = sys.argv[1].upper()
        mode = sys.argv[2].upper()
    except (IndexError, AttributeError):
        raise ValueError("Must select model(anaphora/ner/sentiment) and train/test mode -- e.g: main.py ner test")

    match mode:
        case "TRAIN":
            trainMode = True
        case "TEST":
            trainMode = False
        case _:
            raise ValueError("Must select model(anaphora/ner/sentiment) and train/test mode -- e.g: main.py ner test")




    # Set all random seeds to make experiments reproducible
    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



    # GPU accelerator
    if torch.accelerator.is_available():
        device = torch.cuda.current_device()
        print("Using", torch.cuda.get_device_name(device))
    else:
        device = "cpu"
        print("Using", device)



    # Each task uses a different transformer head and evaluation method,
    # but all tasks are compatible with the shared training loop
    match modelType:
        case "ANAPHORA":
            saveFolder = "Anaphora"

            trainBatch = 12
            valBatch = 4
            testBatch = 32

            trainData, valData, testData = anaphoraData.getData(trainBatch, valBatch, testBatch, numPro, trainMode)
            model = AutoModelForMultipleChoice.from_pretrained("SpanBERT/spanbert-base-cased")
            evaluator = anaEval(device = device)

            if trainMode:
                epochs = 4
                learningRate = 2e-8
                criterion = torch.nn.CrossEntropyLoss()
                optimiser = torch.optim.AdamW(model.parameters(), lr = learningRate, eps=1e-6)
            
        
        
        case "NER":
            saveFolder = "Ner"

            trainBatch = 32
            valBatch = 64
            testBatch = 128
            
            trainData, valData, testData, numLabels, ID2label, label2ID = nerData.getData(trainBatch, valBatch, testBatch, numPro, trainMode)
            model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels = numLabels, id2label = ID2label, label2id = label2ID)
            evaluator = nerEval(device = device, idToLabel=ID2label)

            if trainMode:
                epochs = 4
                learningRate = 2e-5

                criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
                optimiser = torch.optim.AdamW(model.parameters(), lr = learningRate)

            

        case "SENTIMENT":
            saveFolder = "Sentiment"

            trainBatch = 16
            valBatch = 128
            testBatch = 128

            trainData, valData, testData, weights = senitmentData.getData(trainBatch, valBatch, testBatch, numPro, trainMode)
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
            evaluator = sentEval(fitMode = False, device = device, threshold=0.24) # Threshold from best validation

            if trainMode:
                epochs = 3
                learningRate = 1e-5
                dropout = 0.2

                # Increase regularisation by overriding dropout values in the pretrained BERT encoder
                model.dropout = torch.nn.Dropout(dropout)
                for layer in model.bert.encoder.layer:
                    layer.attention.self.dropout.p = dropout
                    layer.attention.output.dropout.p = dropout
                    layer.output.dropout.p = dropout

                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype = torch.float32)).to(device)

                optimiser = torch.optim.AdamW(model.parameters(), lr = learningRate, weight_decay=0.01)

                evaluator = sentEval(fitMode = True, device = device) ## Override for trainmode

        case _:
            raise ValueError("Must select model(anaphora/ner/sentiment) and train/test mode -- e.g: main.py ner test")
        


    
    log = Logger(saveFolder)
    model.to(device)


    if trainMode: 
        trainLoop.trainModel(trainData, valData, model, optimiser, criterion, evaluator, epochs, log, device)
        return


    # Testing uses the best validation checkpoint
    model.load_state_dict(log.loadModel("modelBest.pt"))
    result = testLoop.testModel(testData, model, evaluator, device)
    print(result)


if __name__ == "__main__":
    main()