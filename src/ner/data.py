from itertools import chain
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import datasets


def getData(trainBatch, valBatch, testBatch, numPro, trainMode):
    """
        Prepare named entity recognition datasets.

        Converts word-level entity annotations into token-level labels
        compatible with BERT token classification models.

        The preprocessing pipeline:
        - loads and splits the dataset
        - creates label mappings
        - tokenises input text into BERT subwords
        - aligns entity labels with generated tokens
        - creates PyTorch dataloaders
    """
    dataset = loadDataSet("boltuix/conll2025-ner")

    label2ID , ID2label = findLabelID(dataset)
    numLabels = len(label2ID)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Handles dynamic padding while keeping token-level labels aligned with the padded input sequences.
    dataCollector = DataCollatorForTokenClassification(tokenizer)

    formatInput = formatDataInit(tokenizer, label2ID)

    trainLoader, valLoader = None, None
    if trainMode:
        trainData = dataset["train"].map(formatInput)
        valData = dataset["validation"].map(formatInput)

        trainData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])
        valData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])#

        
        trainLoader = DataLoader(trainData, batch_size=trainBatch, shuffle=True, collate_fn=dataCollector, num_workers=numPro)
        valLoader = DataLoader(valData, batch_size=valBatch, shuffle=True, collate_fn=dataCollector, num_workers=numPro)


    testData = dataset["test"].map(formatInput)

    testData.set_format("torch", columns = ["input_ids",  "attention_mask", "labels"])

    testData = DataLoader(testData, batch_size=testBatch, shuffle=False, collate_fn=dataCollector, num_workers=numPro)

    
    return trainLoader, valLoader, testData, numLabels, ID2label, label2ID


def loadDataSet(name):
    """
        Creates train/validation/test splits.

        The original dataset only provides training data, so a custom split is created:
        - 10% for final testing
        - 20% of remaining data used for validation
    """
    dataset = datasets.load_dataset(name)
    testSplit = dataset["train"].train_test_split(test_size = 0.1)
    trainValDataset = testSplit["train"]
    testDataset = testSplit["test"]

    valSplit = trainValDataset.train_test_split(test_size = 0.2)
    dataset = datasets.DatasetDict({
        "train": valSplit["train"],
        "validation": valSplit["test"],
        "test": testDataset
        
    })

    return dataset


def findLabelID(dataset):
    # Gives labels an id and saves a mapping for both ways of translation.
    # These mappings are needed to configure the classifier head and decode pedictions back to labels.

    # Flatten all entity annotations from every example and find the complete set of labels appearing in the training data.
    labels = list(set(chain.from_iterable(dataset["train"]["ner_tags"])))
    labels.sort()

    label2ID = {}
    ID2label = {}
    for i, label in enumerate(labels):
        label2ID[label] = i
        ID2label[i] = label

    return label2ID , ID2label


def formatDataInit(tokenizer, labelMapping):
    def formatData(dataset):
        tokenInput = tokenizer(dataset["tokens"], is_split_into_words=True)
        
        # BERT uses subword tokenisation while NER annotations are word based. 
        # word_ids maps generated tokens back to their original words so entity
        # labels can be aligned with tokenised input sequnence
        wordID = tokenInput.word_ids()

        labels = []
        preWord = None
        for word in wordID:
            if word is None:
                labels.append(-100)## fills the non eities tokens with -100 so it is ingores by the loss fucntion in training
            elif word != preWord:
                labels.append(labelMapping[dataset["ner_tags"][word]])
            else:
                labels.append(-100) # ingores subword tokens
            preWord = word

        tokenInput["labels"] = labels
        
        return tokenInput
    return formatData