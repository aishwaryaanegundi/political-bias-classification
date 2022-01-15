import torch
import pandas as pd
import emoji
import re
import string
import json 
import glob
import os
import gzip
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from torch import nn

torch.cuda.empty_cache()

path_to_train_data = './data/polly/polly_train.csv'
path_to_test_data = './data/polly/polly_test.csv'
path_to_predicted_files = './data/polly/polly_text_predicted2_gelectra_v1.csv'
path_to_saved_model = './models/gelectra-large-two_extremesv1.pt'
pretrained_model_name = "deepset/gelectra-large"
batch_size = 1

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, normalization=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def preprocess(row):
    result = re.sub(r"http\S+", "", row['Tweet'])
    result = re.sub(r"RT @\S+", "", result)
    result = re.sub(r"@\S+", "", result)
    result = re.sub(r"\n", "", result)
    return result


response_df = pd.read_csv(path_to_test_data)
response_df['text'] = response_df.apply(preprocess, axis=1)
texts = pd.DataFrame()
texts['text'] = response_df['text'].tolist()
texts['labels'] = 0

raw_datasets = Dataset.from_pandas(texts)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
test_data = tokenized_datasets.select(range((texts.shape)[0]))
print(test_data)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)
model.to(device)
model.load_state_dict(torch.load(path_to_saved_model))
model.eval()
prediction_labels = []
softmax_outputs = []
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    softmax_output = nn.functional.softmax(logits, dim=-1)
    for s in softmax_output:
        softmax_outputs.append(s[1].item())
    predictions = torch.argmax(logits, dim=-1)
    for p in predictions:
        prediction_labels.append(p.item())

response_df['prediction'] = prediction_labels
response_df['probabilities'] = softmax_outputs
response_df.to_csv(path_to_predicted_files)
