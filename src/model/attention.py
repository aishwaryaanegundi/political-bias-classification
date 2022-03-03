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
import numpy as np

torch.cuda.empty_cache()

path_to_predicted_data = 'data/polly/polly_train_afd_green_bert_base.csv'
path_to_attentions_data = 'data/polly/polly_train_afd_green_bert_base_attentions.csv'


# path_to_predicted_files = 'data/polly/other_parties_afd_greens_predicted.csv'
path_to_saved_model = 'models/bert-base-afd_green.pt'
pretrained_model_name = "bert-base-german-cased"
batch_size = 8

def get_attentions(sequence):
    tokenized_sequence = tokenizer.tokenize(sequence)
    indexed_tokens = tokenizer.encode(tokenized_sequence, return_tensors='pt')
    outputs = model(indexed_tokens)
    attention_layers = outputs[3]
    layerwise_attentions = []
    for layer in attention_layers:
        for batch_element in layer:
            averages_across_tokens = []
            for head in batch_element:
                attention_map = head.cpu().detach().numpy()
                average_attention_across_tokens = np.mean(attention_map, axis=0)
                averages_across_tokens.append(average_attention_across_tokens)
            averages_across_tokens_array = np.asarray(averages_across_tokens)
            average_across_heads = np.mean(averages_across_tokens_array, axis=0)
            layerwise_attentions.append(average_across_heads)
    average_across_layers = np.mean(np.asarray(layerwise_attentions), axis=0)
    return tokenized_sequence, layerwise_attentions, average_across_layers.tolist()

## obtaining attentions
from transformers import BertModel, BertConfig, BertTokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True, output_attentions=True)
model = BertModel.from_pretrained(path_to_saved_model, config=config)
# sequence = "Natürlich braucht die #Verkehrswende mehr. Unser Programm hat ja auch mehr."
# tokenized_sequence, layerwise_attentions, average_across_layers = get_attentions(sequence)
# print(tokenized_sequence)
# print(average_across_layers)
# print(layerwise_attentions)
def strip(row):
    text = row['text']
    return text.strip()
predicted = pd.read_csv(path_to_predicted_data)
predicted['text'] = predicted.apply(strip, axis=1)
predicted = predicted.replace('', np.nan, regex=True)
predicted = predicted.dropna()
print(predicted.shape)
tokens = []
layerwise_attentions = []
average_across_layers = []
for index, row in predicted.iterrows():
    text = row['text']
    tokenized_sequence, l_a, a_a_l = get_attentions(text)
    print(a_a_l)
    tokens.append(tokenized_sequence)
    layerwise_attentions.append(l_a)
    average_across_layers.append(a_a_l)
predicted['tokens'] = tokens
predicted['layerwise_attentions'] = layerwise_attentions
predicted['average_across_layers'] = average_across_layers
predicted.to_csv(path_to_attentions_data)

## Obtaining attributions
# from transformers_interpret import SequenceClassificationExplainer
# model_name = "bert-base-german-cased"
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print('device: ', device)
# model.to(device)
# model.load_state_dict(torch.load(path_to_saved_model))
# model.eval()

# sequence = "Natürlich braucht die #Verkehrswende mehr. Unser Programm hat ja auch mehr."
# cls_explainer = SequenceClassificationExplainer(model, tokenizer, embedding_type=1)
# predicted = pd.read_csv(path_to_predicted_data)
# attributions = []
# classes = []
# for index, row in predicted.iterrows():
#     text = row['text']
#     word_attributions = cls_explainer(text)
#     classes.append(cls_explainer.predicted_class_index)
#     attributions.append(word_attributions)
# predicted['attributions'] = attributions
# predicted['attribute_class'] = classes
# predicted.to_csv(path_to_attributions_data)