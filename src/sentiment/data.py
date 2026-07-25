from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import numpy as np

def getData(trainBatch, valBatch, testBatch, numPro, trainMode):
    """
        Prepare sentiment classification datasets.

        Loads the TweetEval offensive language dataset, tokenises the text
        using the BERT tokenizer, converts labels into the format expected by
        PyTorch training, and returns DataLoaders for training, validation,
        and testing.

        Training and validation loaders are only created during training mode
        as they are not required during final inference.
    """

    dataset = datasets.load_dataset("cardiffnlp/tweet_eval", "offensive")


    # Calculate weighted loss values to compensate for class imbalance.
    # Offensive samples are underrepresented, so the positive class receives a larger contribution during CrossEntropyLoss.
    # The *5 scaling is used to weigh the offensive penalty more as a false negative is worse than a false positive.
    total = len(dataset["train"]["label"])
    offensive = np.sum(dataset["train"]["label"])
    nonOffensive  = total - offensive
    weights = [total/(2*nonOffensive ), (total/(2*offensive)) * 5]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    formatInput = formatDataINIT(tokenizer)


    trainLoader, valLoader = None, None
    if trainMode:
        trainData = dataset["train"].map(formatInput, batched = True)
        valData = dataset["validation"].map(formatInput, batched = True)

        # Rename labels to match the argument name expected by training loop when batches called
        trainData = trainData.rename_column("label", "labels")
        valData = valData.rename_column("label", "labels")

        # Convert selected dataset columns into PyTorch tensors so they can be directly consumed by DataLoader.
        trainData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])
        valData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])
        
        trainLoader = DataLoader(trainData, batch_size=trainBatch, shuffle=True, num_workers=numPro)
        valLoader = DataLoader(valData, batch_size=valBatch, shuffle=True, num_workers=numPro)



    testData = dataset["test"].map(formatInput, batched = True)

    testData = testData.rename_column("label", "labels")

    testData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])

    # Test data is kept in a fixed order because no optimisation occurs and reproducible evaluation is preferred over random sampling.
    testLoader = DataLoader(testData, batch_size = testBatch, shuffle=False, num_workers=numPro)


    return trainLoader, valLoader, testLoader, weights

    

def formatDataINIT(tokenizer):
    """
        Creates a tokenisation function using the provided tokenizer.

        Returning a function allows the Hugging Face dataset.map API to apply
        the same preprocessing pipeline efficiently across the dataset.
    """
    def formatData(dataset):
        tokenInput = tokenizer(dataset["text"], padding = "max_length", truncation = True, max_length = 128)
        return tokenInput
    return formatData


