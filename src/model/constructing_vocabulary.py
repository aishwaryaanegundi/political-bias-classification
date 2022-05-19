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
import ast
import numpy as np
from icu_tokenizer import Tokenizer


torch.cuda.empty_cache()
afd_green = 'afd-green'
fdp_linke = 'fdp_linke'
dimension = afd_green
path_to_train_data = 'data/polly/polly_train_'+dimension+'_bert_base.csv'
path_to_store_vocabulary = 'data/polly/'+dimension+'-train-vocabulary.csv'
path_to_store_pmi_tfidf_vocabulary = 'data/polly/'
path_to_test_data = 'data/polly/polly_test_'+'afd_green'+'_bert_base_predicted.csv'
# path_to_predicted_files = 'data/polly/other_parties_afd_greens_predicted.csv'
path_to_store_results = 'results/error_analysis'
path_to_saved_model = 'models/bert-base-'+dimension+'.pt'
pretrained_model_name = "bert-base-german-cased"
german_stop_words = stopwords.words('german')
tokenizer = Tokenizer(lang='de')

# ## obtaining attentions
# def get_attentions(sequence):
#     tokenized_sequence = tokenizer.tokenize(sequence)
#     indexed_tokens = tokenizer.encode(tokenized_sequence, return_tensors='pt')
#     outputs = model(indexed_tokens)
#     attention_layers = outputs[3]
#     layerwise_attentions = []
#     for layer in attention_layers:
#         for batch_element in layer:
#             averages_across_tokens = []
#             for head in batch_element:
#                 attention_map = head.cpu().detach().numpy()
#                 average_attention_across_tokens = np.mean(attention_map, axis=0)
#                 averages_across_tokens.append(average_attention_across_tokens)
#             averages_across_tokens_array = np.asarray(averages_across_tokens)
#             average_across_heads = np.mean(averages_across_tokens_array, axis=0)
#             layerwise_attentions.append(average_across_heads)
#     average_across_layers = np.mean(np.asarray(layerwise_attentions), axis=0)
#     return tokenized_sequence, layerwise_attentions, average_across_layers.tolist()

# def strip(row):
#     text = row['text']
#     return text.strip()

# def combine_subword_scores(tokenized_sequence, l_a, a_a_l):
#     print(tokenized_sequence)
#     l_a = np.asarray(l_a)
#     l_a = l_a[:,1:-1]
#     print(len(tokenized_sequence), l_a.shape, len(a_a_l))
#     new_seq = []
#     c = 0
#     for i,v in enumerate(tokenized_sequence):
#         if v.startswith('##'):
#             print('i,v: ', i, v, c)
#             v = v.replace("##", "")
#             new_seq[i-c-1] = new_seq[i-c-1] + v
#             sum_ = l_a[:,i-c-1] + l_a[:,i-c]
#             l_a = np.concatenate((l_a[:,:i-c-1],sum_[:,None],l_a[:,i-c+1:]), axis=1)
#             print('shape', l_a.shape)
#             c = c + 1
#         else:
#             new_seq.append(v)
#         print(new_seq)
#         print(l_a)
#     new_a_a_l = np.mean(np.asarray(l_a), axis=0)
#     print(len(new_seq), l_a.shape, new_a_a_l.shape)
#     return new_seq, l_a, new_a_a_l

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

# # Computing tfidf scores
# def get_top_n_features_given_list_of_tweet_texts(n, tweets_left, tweets_right, left=True):
#     left_document = ' '.join(tweets_left)
#     left_length = len(left_document.split())
#     right_document = ' '.join(tweets_right)
#     right_length = len(right_document.split())
#     cv = CountVectorizer(stop_words=german_stop_words)
#     cv_fit = cv.fit_transform([left_document, right_document])
#     features = cv.get_feature_names()
#     counts = cv_fit.toarray()
#     print(counts.shape)
#     tfidf = TfidfVectorizer(stop_words=german_stop_words)
    
#     x = tfidf.fit_transform([left_document, right_document])
#     idfs = tfidf.idf_
#     if left:
#         scores = np.multiply(counts[0]/left_length, idfs)
#         frequencies = counts[0]/len(tweets_left)
#         feature_score = []
#         for i in range(len(features)):
#             feature_score.append({'feature':features[i],'score': scores[i], 'frequency': frequencies[i]})
#         tfidf_df = pd.DataFrame(feature_score)
#         tfidf_df = tfidf_df.sort_values(by=['score'], ascending=False)
#         top_n = tfidf_df.head(len(features))
#         return top_n
#     else:
#         scores = np.multiply(counts[1]/right_length, idfs)
#         frequencies = counts[1]/len(tweets_right)
#         feature_score = []
#         for i in range(len(features)):
#             feature_score.append({'feature':features[i],'score': scores[i],'frequency': frequencies[i]})
#         tfidf_df = pd.DataFrame(feature_score)
#         tfidf_df = tfidf_df.sort_values(by=['score'], ascending=False)
#         top_n = tfidf_df.head(len(features))
#         return top_n
    
# tfidf_vocabulary = pd.DataFrame()
# n=500
# train_data = pd.read_csv(path_to_train_data)
# top_tfidf_left = get_top_n_features_given_list_of_tweet_texts(n,
#                             train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist())
# top_tfidf_right = get_top_n_features_given_list_of_tweet_texts(n,
#                             train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(), left=False)

# tfidf_vocabulary['tf_idf_left_keys'] = top_tfidf_left['feature'].tolist()
# tfidf_vocabulary['tf_idf_left_scores'] = top_tfidf_left['score'].tolist()
# tfidf_vocabulary['tf_idf_left_frequencies'] = top_tfidf_left['frequency'].tolist()
# tfidf_vocabulary['tf_idf_left_rank'] = tfidf_vocabulary['tf_idf_left_scores'].rank()

# tfidf_vocabulary['tf_idf_right_keys'] = top_tfidf_right['feature'].tolist()
# tfidf_vocabulary['tf_idf_right_scores'] = top_tfidf_right['score'].tolist()
# tfidf_vocabulary['tf_idf_right_frequencies'] = top_tfidf_right['frequency'].tolist()
# tfidf_vocabulary['tf_idf_right_rank'] = tfidf_vocabulary['tf_idf_right_scores'].rank()

# # Computing pmi scores
# def get_top_n_features_based_pmi(n, left_tweets, right_tweets, left=True):
#     n_left = len(left_tweets)
#     n_right = len(right_tweets)
#     left_document = ' '.join(left_tweets)
#     right_document = ' '.join(right_tweets)
#     document = ' '.join([left_document,right_document])
#     cv = CountVectorizer(stop_words=german_stop_words)
#     cv_fit = cv.fit_transform([left_document, right_document])
#     features = cv.get_feature_names()
    
#     counts = cv_fit.toarray()
#     pmi = []
#     for index, f in enumerate(features):
#         N = (n_left + n_right)
#         p_x = (counts[0][index]+counts[1][index])/N
#         p_x_left = counts[0][index]/N
#         p_x_right = counts[1][index]/N
#         p_left = n_left/N
#         p_right = n_right/N
        
#         try:
#             pmi_x_left = math.log2(p_x_left/(p_x * p_left))
#         except ZeroDivisionError:
#             pmi_x_left = 0
#         except Exception as e:
#             pmi_x_left = 0
#         try:
#             pmi_x_right = math.log2(p_x_right/(p_x * p_right))
#         except ZeroDivisionError:
#             pmi_x_right = 0
#         except Exception as e:
#             pmi_x_right = 0
        
#         combined_score = (pmi_x_left - pmi_x_right)
#         if left:
#             combined_score = (pmi_x_left - pmi_x_right) * p_x_left
#         else:
#             combined_score = (pmi_x_right - pmi_x_left) * p_x_right
            
#         pmi.append({'feature':f,
#                     'pmi_left': pmi_x_left,
#                     'pmi_right': pmi_x_right,
#                     'score': combined_score})
    
#     pmi_df = pd.DataFrame(pmi)
#     pmi_df = pmi_df.sort_values(by=['score'], ascending=False)
#     top_n = pmi_df.head(len(features))
#     return top_n

# pmi_vocabulary = pd.DataFrame()
# train_data = pd.read_csv(path_to_train_data)
# top_pmi_left = get_top_n_features_based_pmi(n,
#                              train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(),
#                             left=True)
# top_pmi_right = get_top_n_features_based_pmi(n,
#                              train_data[train_data['labels']==0]['text'].tolist(),
#                              train_data[train_data['labels']==1]['text'].tolist(), left=False)


# pmi_vocabulary['pmi_left_keys'] = top_pmi_left['feature'].tolist()
# pmi_vocabulary['pmi_left_scores'] = top_pmi_left['score'].tolist()
# pmi_vocabulary['pmi_left_rank'] = pmi_vocabulary['pmi_left_scores'].rank()
# pmi_vocabulary['pmi_right_keys'] = top_pmi_right['feature'].tolist()
# pmi_vocabulary['pmi_right_scores'] = top_pmi_right['score'].tolist()
# pmi_vocabulary['pmi_right_rank'] = pmi_vocabulary['pmi_right_scores'].rank()



# tfidf_vocabulary.to_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-tfidf.csv')
# pmi_vocabulary.to_csv(path_to_store_pmi_tfidf_vocabulary + dimension+'-train-pmi.csv')
                      
# # Constructing the vocabulary important token based on frequency of top two tokens in each tweet of train data
# attention_data = pd.read_csv(path_to_store_vocabulary)
# def preprocess_attention_scores(row):
#     scores = row['average_across_layers']   
#     new_scores = []
#     scores = scores.split()
#     for s in scores:
#         s = s.rstrip()
#         s = s.removesuffix(']')
#         s = s.removeprefix('[') 
#         new_scores.append(s)
#     return new_scores

# attention_data['attention_scores'] = attention_data.apply(preprocess_attention_scores, axis=1)
# attention_data.to_csv(path_to_store_pmi_tfidf_vocabulary + dimension+'-train-attention.csv')

# def get_top_two_tokens(row):
#     scores = row['attention_scores']
#     tokens=row['tokens']
#     try:
#         if isinstance(scores,str):
#             scores = ast.literal_eval(scores)
#         if isinstance(tokens,str):
#             tokens = ast.literal_eval(tokens)
#         tokens = np.asarray(tokens)
#         scores = np.asarray(scores)
#         ind = np.argpartition(scores, -2)[-2:]
#         return list(tokens[ind])
#     except Exception as e:
#         print(repr(e))
#         return None
        
# attention_vocabulary = pd.read_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-attention.csv')
# attention_vocabulary['top_two_tokens'] = attention_vocabulary.apply(get_top_two_tokens, axis=1)
# attention_vocabulary.to_csv(path_to_store_pmi_tfidf_vocabulary + dimension+'-train-attention.csv')

# attention_vocabulary = attention_vocabulary.dropna()
# left = attention_vocabulary[attention_vocabulary['labels']==0]
# all_important_attention_tokens = []
# for i,r in left.iterrows():
#     tokens = r['top_two_tokens']
#     if isinstance(tokens,str):
#         tokens = ast.literal_eval(tokens)
#     all_important_attention_tokens = all_important_attention_tokens + tokens
# token_frequencies = Counter(all_important_attention_tokens)
# token_frequencies = pd.DataFrame.from_dict(token_frequencies, orient='index').reset_index()
# token_frequencies = token_frequencies.rename(columns={'index':'token', 0:'frequency'})
# token_frequencies['rank'] = token_frequencies['frequency'].rank()
# token_frequencies.to_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-attention-frequencies-left.csv')
# right = attention_vocabulary[attention_vocabulary['labels']==1]
# all_important_attention_tokens = []
# for i,r in right.iterrows():
#     tokens = r['top_two_tokens']
#     if isinstance(tokens,str):
#         tokens = ast.literal_eval(tokens)
#     all_important_attention_tokens = all_important_attention_tokens + tokens
# token_frequencies = Counter(all_important_attention_tokens)
# token_frequencies = pd.DataFrame.from_dict(token_frequencies, orient='index').reset_index()
# token_frequencies = token_frequencies.rename(columns={'index':'token', 0:'frequency'})
# token_frequencies['rank'] = token_frequencies['frequency'].rank()
# token_frequencies.to_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-attention-frequencies-right.csv')

# Error Analysis
tfidf_vocabulary = pd.read_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-tfidf.csv')
pmi_vocabulary = pd.read_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-pmi.csv')
attention_vocabulary_left = pd.read_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-attention-frequencies-left.csv')
attention_vocabulary_right = pd.read_csv(path_to_store_pmi_tfidf_vocabulary +dimension+'-train-attention-frequencies-right.csv')
test_data = pd.read_csv(path_to_test_data)

def get_diff_of_importance_score(row,
                                 right_keys,
                                 right_scores,
                                 left_keys,
                                 left_scores,
                                 right_frequencies =[],
                                 left_frequencies=[], tfidf=False):
    text = row['text']
#     text = text.split()
    text = tokenizer.tokenize(text)
    labels = row['labels']
    diffs = []
    for word in text:
        word = word.lower()
        print('Word is: ', word)
        if word not in german_stop_words:
            print(word, ' is not in german stop words')
            try:
                r_sel = np.where(right_keys==word)
                print('r_sel: ', r_sel)
                r_score = right_scores[r_sel]
                print('r_score: ', r_score)
                l_sel = np.where(left_keys==word)
                print('l_sel: ', l_sel)
                l_score = left_scores[l_sel]
                print('l_score: ', l_score)
                if labels == 0:
                    diff = r_score - l_score
                    if tfidf:
                        r_freq = right_frequencies[r_sel]
                        print('r_freq: ', r_freq)
                        diff = diff * r_freq
                else:
                    diff = l_score - r_score
                    if tfidf:
                        l_freq = lrft_frequencies[r_sel]
                        print('l_freq: ', l_freq)
                        diff = diff * l_freq
            except Exception as e:
                print('Exception occured: ', repr(e))
                diff = 0
        else:
            print(word, ' is in german stop words')
            diff = 0
            
        if not diff:
            print('diff was not defined')
            diff = 0
        if diff == 0:
            print('appending diff =0 ')
            diffs.append({'token': word,'diff': diff})
        else:
            print(word, diff)
            diffs.append({'token': word, 'diff': diff[0]})
#     print(diffs)
    return diffs

## calculating tf idf diffs
# test_data['tf_idf_diffs'] = test_data.apply(get_diff_of_importance_score,
#                                            right_keys = np.asarray(tfidf_vocabulary['tf_idf_right_keys'].tolist()),
#                                            right_scores = np.asarray(tfidf_vocabulary['tf_idf_right_scores'].tolist()),
#                                            left_keys = np.asarray(tfidf_vocabulary['tf_idf_left_keys'].tolist()),
#                                            left_scores = np.asarray(tfidf_vocabulary['tf_idf_left_scores'].tolist()),
#                                            right_frequencies = np.asarray(tfidf_vocabulary['tf_idf_right_frequencies'].tolist()),
#                                            left_frequencies = np.asarray(tfidf_vocabulary['tf_idf_left_frequencies'].tolist()),
#                                            tfidf=True,
#                                            axis=1)
# test_data.to_csv(path_to_store_results+'/ea_'+dimension+'.csv')

# Calculating pmi diffs
test_data['pmi_diffs'] = test_data.apply(get_diff_of_importance_score,
                                           right_keys = np.asarray(pmi_vocabulary['pmi_right_keys'].tolist()),
                                           right_scores = np.asarray(pmi_vocabulary['pmi_right_scores'].tolist()),
                                           left_keys = np.asarray(pmi_vocabulary['pmi_left_keys'].tolist()),
                                           left_scores = np.asarray(pmi_vocabulary['pmi_left_scores'].tolist()),
                                           tfidf=False,
                                           axis=1)
test_data.to_csv(path_to_store_results+'/ea_pmi'+dimension+'.csv')
# Correlation of attention with tfidf and pmi
# Obtain the attention scores for test data
# tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
# config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True, output_attentions=True)
# model = BertModel.from_pretrained(path_to_saved_model, config=config)
# test_data = pd.read_csv(path_to_test_data)
# test_data['text'] = test_data.apply(strip, axis=1)
# test_data = test_data.replace('', np.nan, regex=True)
# test_data = test_data.dropna()
# tokens = []
# layerwise_attentions = []
# average_across_layers = []
# for index, row in test_data.iterrows():
#     text = row['text']
#     tokenized_sequence, l_a, a_a_l = get_attentions(text)
#     tokenized_sequence, l_a, a_a_l = combine_subword_scores(tokenized_sequence, l_a, a_a_l)
#     tokens.append(tokenized_sequence)
#     layerwise_attentions.append(l_a)
#     average_across_layers.append(a_a_l)
    
# test_data['tokens'] = tokens
# test_data['layerwise_attentions'] = layerwise_attentions
# test_data['average_across_layers'] = average_across_layers
# test_data.to_csv(path_to_store_results+dimension+'-test-data-attentions.csv')
# test_data = pd.read_csv(path_to_store_results+dimension+'-test-data-attentions.csv')
# test_data['attention_scores'] = test_data.apply(preprocess_attention_scores, axis=1)
# test_data.to_csv(path_to_store_results+dimension+'-test-data-attentions.csv')