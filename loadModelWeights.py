from huggingface_hub import hf_hub_download
from pathlib import Path

"""
    Loads model weights from the Hugging Face repo    
"""



repo = "bengoodhew/transformer-nlp-models-3in1"



downloaded = hf_hub_download(repo_id=repo, filename="anaphora.pt")

savePath = "experiments/Anaphora/models"
Path(savePath).mkdir(parents=True, exist_ok=True)
Path(downloaded).replace(savePath+"/modelBest.pt")





downloaded = hf_hub_download(repo_id=repo, filename="ner.pt")

savePath = "experiments/Ner/models"
Path(savePath).mkdir(parents=True, exist_ok=True)
Path(downloaded).replace(savePath+"/modelBest.pt")




downloaded = hf_hub_download(repo_id=repo, filename="sentiment.pt")

savePath = "experiments/Sentiment/models"
Path(savePath).mkdir(parents=True, exist_ok=True)
Path(downloaded).replace(savePath+"/modelBest.pt")

