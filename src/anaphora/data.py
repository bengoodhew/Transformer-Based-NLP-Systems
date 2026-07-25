import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import AutoTokenizer
import datasets

from anaphora import preprocess

def getData(trainBatch, valBatch, testBatch, numPro, trainMode):
    """
        Prepare anaphora resolution datasets.

        The original PreCo dataset contains coreference chains between
        mentions in documents. This preprocessing converts those chains
        into a multiple-choice classification task suitable for SpanBERT.

        Each example contains:
        - a text span containing a pronoun
        - three candidate antecedents
        - the index of the correct candidate

        The model is trained to select which candidate the pronoun refers to.
    """

    dataset = datasets.load_dataset("coref-data/preco_indiscrim")
    
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

    # Custom collator is required because this task uses multiple-choice inputs.
    # Each example contains three separately tokenised candidate sequences
    # which may have different lengths and require padding at batch creation.
    dataCollector = makeDataCollator(tokenizer)

    # spaCy is used during preprocessing to identify noun and pronoun mentions.
    # NER and parsing are disabled because only part-of-speech information is required.
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    
    # Creates the preprocessing pipeline that:
    # - extracts valid noun-pronoun coreference pairs
    # - creates distractor candidates
    # - extracts the relevant text span
    # - tokenises candidate answers for SpanBERT
    dataFormatter = preprocess.makeDataFormatter(nlp, tokenizer)

    trainLoader, valLoader = None, None
    if trainMode:
        # Dataset.map applies the expensive preprocessing step.
        # Multiple processes are used because preprocessing involves document-level processing and tokenisation.
        trainData = dataset["train"].map(dataFormatter, num_proc = numPro)
        valData = dataset["validation"].map(dataFormatter, num_proc = numPro)


        trainData = datasetForModel(trainData)
        valData = datasetForModel(valData)

        trainLoader = DataLoader(trainData, batch_size=trainBatch, shuffle=True, collate_fn=dataCollector)
        valLoader = DataLoader(valData, batch_size=valBatch, shuffle=True, collate_fn=dataCollector)

    # Test data follows the same preprocessing pipeline but does not require training/validation loaders.
    testData = dataset["test"].map(dataFormatter, num_proc = numPro)

    testData = datasetForModel(testData)
    
    testLoader = DataLoader(testData, batch_size=testBatch, shuffle=False, collate_fn=dataCollector)

    return trainLoader, valLoader, testLoader


class datasetForModel(Dataset):
    """
        Converts processed Hugging Face dataset into PyTorch dataset format.

        Invalid examples are removed because some documents cannot produce a clean noun-pronoun relationship.
    """
    def __init__(self, dataset):
        self.data = []
        for data in dataset:
            # Remove examples where preprocessing failed.
            # Very large spans are also removed because truncating them could
            # remove important context needed for resolving the pronoun.
            if data["input_ids"] == None or len(data["span"]) > 500:
                continue

            self.data.append({"input_ids": data["input_ids"],
                "attention_masks": data["attention_masks"],
                "label": data["label"],
                "span": data["span"], "candidates": data["candidates"], "pronoun": data["pronoun"]})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]



## pads data per batch to longest sequnce in batch
def makeDataCollator(tokenizer):
    """
        Creates a batch collator for the multiple-choice anaphora task.

        Unlike standard classification tasks, each example contains multiple
        candidate sequences. Each candidate sequence is padded dynamically
        to the longest sequence within the current batch.
    """
    def collateData(batch):
        returnBatch = {"input_ids": [],"attention_mask": [],"labels": []}

        numChoices = len(batch[0]["attention_masks"])

        # Find the longest candidate sequence in the batch so padding is only applied where required.
        maxSeqLen = 0

        for item in batch:
            for i in range(numChoices):
                if maxSeqLen < len(item["attention_masks"][i]):
                    maxSeqLen = len(item["attention_masks"][i])

        for item in batch:
            # Pad each candidate independently so all choices have the same sequence length before being converted into tensors.
            for i in range(numChoices):
                paddingSize = maxSeqLen - len(item["attention_masks"][i])
                for _ in range(paddingSize):
                    item["input_ids"][i].append(tokenizer.pad_token_id)
                    item["attention_masks"][i].append(0)


            returnBatch["input_ids"].append(item["input_ids"])
            returnBatch["attention_mask"].append(item["attention_masks"])
            returnBatch["labels"].append(item["label"])


        returnBatch["input_ids"] = torch.tensor(returnBatch["input_ids"], dtype=torch.long)
        returnBatch["attention_mask"] = torch.tensor(returnBatch["attention_mask"], dtype=torch.long)
        returnBatch["labels"] = torch.tensor(returnBatch["labels"], dtype=torch.long)
        return returnBatch
    return collateData
