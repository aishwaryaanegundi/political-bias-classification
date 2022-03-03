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
from transformers import BertModel, BertConfig, BertTokenizer
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import font_manager
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from nltk.corpus import stopwords


torch.cuda.empty_cache()

path_to_train_data = 'data/polly/polly_train_afd_green_bert_base.csv'
path_to_store_vocabulary = 'data/polly/afd-green-train-vocabulary.csv'
path_to_store_pmi_tfidf_vocabulary = 'data/polly/afd-green-train-pmi_tfidf.csv'
# path_to_predicted_files = 'data/polly/other_parties_afd_greens_predicted.csv'

path_to_saved_model = 'models/bert-base-afd_green.pt'
pretrained_model_name = "bert-base-german-cased"
german_stop_words = stopwords.words('german')


## obtaining attentions
def get_attentions(sequence):
#     print('sequence: ', sequence)
    tokenized_sequence = tokenizer.tokenize(sequence)
    indexed_tokens = tokenizer.encode(tokenized_sequence, return_tensors='pt')
    outputs = model(indexed_tokens)
    attention_layers = outputs[3]
#     print('attention shape: ',attention_layers)
#     print('attention shape: ', outputs[3][0].shape)
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

def strip(row):
    text = row['text']
    return text.strip()

def combine_subword_scores(tokenized_sequence, l_a, a_a_l):
    print(tokenized_sequence)
    l_a = np.asarray(l_a)
    l_a = l_a[:,1:-1]
    print(len(tokenized_sequence), l_a.shape, len(a_a_l))
    new_seq = []
    c = 0
    for i,v in enumerate(tokenized_sequence):
        if v.startswith('##'):
            print('i,v: ', i, v, c)
            v = v.replace("##", "")
            new_seq[i-c-1] = new_seq[i-c-1] + v
            sum_ = l_a[:,i-c-1] + l_a[:,i-c]
            l_a = np.concatenate((l_a[:,:i-c-1],sum_[:,None],l_a[:,i-c+1:]), axis=1)
            print('shape', l_a.shape)
            c = c + 1
        else:
            new_seq.append(v)
        print(new_seq)
        print(l_a)
    new_a_a_l = np.mean(np.asarray(l_a), axis=0)
    print(len(new_seq), l_a.shape, new_a_a_l.shape)
    return new_seq, l_a, new_a_a_l

# tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
# config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True, output_attentions=True)
# model = BertModel.from_pretrained(path_to_saved_model, config=config)
# data = pd.read_csv(path_to_train_data)
# data['text'] = data.apply(strip, axis=1)
# data = data.replace('', np.nan, regex=True)
# data = data.dropna()
# tokens = []
# layerwise_attentions = []
# average_across_layers = []
# for index, row in data.iterrows():
#     text = row['text']
#     tokenized_sequence, l_a, a_a_l = get_attentions(text)
#     tokenized_sequence, l_a, a_a_l = combine_subword_scores(tokenized_sequence, l_a, a_a_l)
#     tokens.append(tokenized_sequence)
#     layerwise_attentions.append(l_a)
#     average_across_layers.append(a_a_l)
    
# data['tokens'] = tokens
# data['layerwise_attentions'] = layerwise_attentions
# data['average_across_layers'] = average_across_layers
# data.to_csv(path_to_store_vocabulary)

# Computing tfidf scores
def get_top_n_features_given_list_of_tweet_texts(n, hashtags_left, hashtags_right, left=True):
    left_document = ' '.join(hashtags_left)
    left_length = len(left_document.split())
    right_document = ' '.join(hashtags_right)
    right_length = len(right_document.split())
    cv = CountVectorizer(stop_words=german_stop_words)
    cv_fit = cv.fit_transform([left_document, right_document])
    features = cv.get_feature_names()
    counts = cv_fit.toarray()
    print(counts.shape)
    tfidf = TfidfVectorizer(stop_words=german_stop_words)
    
    x = tfidf.fit_transform([left_document, right_document])
    idfs = tfidf.idf_
    if left:
        scores = np.multiply(counts[0]/left_length, idfs)
        frequencies = counts[0]/len(hashtags_left)
        feature_score = []
        for i in range(len(features)):
            feature_score.append({'feature':features[i],'score': scores[i]})
        tfidf_df = pd.DataFrame(feature_score)
        tfidf_df = tfidf_df.sort_values(by=['score'], ascending=False)
        top_n = tfidf_df.head(n)
        return top_n
    else:
        scores = np.multiply(counts[1]/right_length, idfs)
        frequencies = counts[1]/len(hashtags_right)
        feature_score = []
        for i in range(len(features)):
            feature_score.append({'feature':features[i],'score': scores[i]})
        tfidf_df = pd.DataFrame(feature_score)
        tfidf_df = tfidf_df.sort_values(by=['score'], ascending=False)
        top_n = tfidf_df.head(n)
        return top_n
    
pmi_tf_vocabulary = pd.DataFrame()
n=500
# train_data = pd.read_csv(path_to_train_data)
# top_tfidf_left = get_top_n_features_given_list_of_tweet_texts(n,
#                             train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist())
# top_tfidf_right = get_top_n_features_given_list_of_tweet_texts(n,
#                             train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(), left=False)

# pmi_tf_vocabulary['tf_idf_left_keys'] = top_tfidf_left['feature'].tolist()
# pmi_tf_vocabulary['tf_idf_left_scores'] = top_tfidf_left['score'].tolist()
# pmi_tf_vocabulary['tf_idf_right_keys'] = top_tfidf_right['feature'].tolist()
# pmi_tf_vocabulary['tf_idf_right_scores'] = top_tfidf_right['score'].tolist()

# Computing pmi scores
def get_top_n_features_based_pmi(n, left_tweets, right_tweets, left=True):
    n_left = len(left_tweets)
    n_right = len(right_tweets)
    left_document = ' '.join(left_tweets)
    right_document = ' '.join(right_tweets)
    document = ' '.join([left_document,right_document])
    cv = CountVectorizer(stop_words=german_stop_words)
    cv_fit = cv.fit_transform([left_document, right_document])
    features = cv.get_feature_names()
    
    counts = cv_fit.toarray()
    pmi = []
    for index, f in enumerate(features):
        N = (n_left + n_right)
        p_x = (counts[0][index]+counts[1][index])/N
        p_x_left = counts[0][index]/N
        p_x_right = counts[1][index]/N
        p_left = n_left/N
        p_right = n_right/N
        
        try:
            pmi_x_left = math.log2(p_x_left/(p_x * p_left))
        except ZeroDivisionError:
            pmi_x_left = 0
        except Exception as e:
            pmi_x_left = 0
        try:
            pmi_x_right = math.log2(p_x_right/(p_x * p_right))
        except ZeroDivisionError:
            pmi_x_right = 0
        except Exception as e:
            pmi_x_right = 0
        
        combined_score = (pmi_x_left - pmi_x_right)
        if left:
            combined_score = (pmi_x_left - pmi_x_right) * p_x_left
        else:
            combined_score = (pmi_x_right - pmi_x_left) * p_x_right
            
        pmi.append({'feature':f,
                    'pmi_left': pmi_x_left,
                    'pmi_right': pmi_x_right,
                    'score': combined_score})
    
    pmi_df = pd.DataFrame(pmi)
    pmi_df = pmi_df.sort_values(by=['score'], ascending=False)
    top_n = pmi_df.head(n)
    return top_n

# train_data = pd.read_csv(path_to_train_data)
# top_pmi_left = get_top_n_features_based_pmi(n,
#                              train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(),
#                             left=True)
# top_pmi_right = get_top_n_features_based_pmi(n,
#                              train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(), left=False)


# pmi_tf_vocabulary['pmi_left_keys'] = top_pmi_left['feature'].tolist()
# pmi_tf_vocabulary['pmi_left_scores'] = top_pmi_left['score'].tolist()
# pmi_tf_vocabulary['pmi_right_keys'] = top_pmi_right['feature'].tolist()
# pmi_tf_vocabulary['pmi_right_scores'] = top_pmi_right['score'].tolist()

# pmi_tf_vocabulary.to_csv(path_to_store_pmi_tfidf_vocabulary)

# Error Analysis
pmi_tf_vocabulary = pd.read_csv(path_to_store_pmi_tfidf_vocabulary)
attention_data = pd.read_csv(path_to_store_vocabulary)


