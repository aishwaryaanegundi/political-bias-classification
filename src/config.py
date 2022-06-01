import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizerFast


class Config:
    DATA_DIR = os.path.abspath("data")
    DATASET_POLLY_PATH = os.path.join(DATA_DIR, "POLLY.xlsx")
    DATASET_POLLY_CLASSIFIED_PATH = os.path.join(DATA_DIR, "polly_classified.jsonl")
    DOCKER_COMPOSE_SERVICE_NAME = "pbgcs"
    DOCKER_IMAGE = "konstantinschulz/political-bias-german-cultural-socioeconomic:v1"
    DOCKER_PORT_CREDIBILITY = 8000
    HOST_PORT_CREDIBILITY = 8181
    LANGUAGE_SERVICE: str = "political-bias-german-cultural-socioeconomic"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir: str = os.path.abspath("models")
    model_name: str = os.path.join(models_dir, "bert-base-german-cased")
    model_path_cultural: str = os.path.join(models_dir, "bert-base-afd_green.pt")
    model_path_socioeconomic: str = os.path.join(models_dir, "bert-base-fdp_linke.pt")
    model_cultural: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)
    model_socioeconomic: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)

    def load_ckp(checkpoint_fpath, model) -> None:
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into       
        """
        # load check point; transfer it to CPU, no matter on which device it was originally created
        checkpoint = torch.load(checkpoint_fpath, map_location="cpu")
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint)


models: list[tuple[BertForSequenceClassification, str]] = [(Config.model_cultural, Config.model_path_cultural),
                                                           (Config.model_socioeconomic, Config.model_path_socioeconomic)]
for model, path in models:
    Config.load_ckp(path, model)
