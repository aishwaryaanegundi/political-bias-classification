from gensim.models import KeyedVectors
import pandas as pd
from gensim.test.utils import datapath
import gensim
import numpy as np
from icu_tokenizer import Tokenizer
from icu_tokenizer import Normalizer
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 
import re
import emoji

afd_green = 'afd_green'
fdp_linke = 'fdp_linke'
dimension = afd_green

cap_path = datapath("/netscratch/aishwarya/political-bias-classification/data/fasttext/cc.de.300.bin")
path_to_train_data = 'data/polly/polly_train_' + dimension + '_bert_base.csv'
path_to_test_data = 'data/polly/polly_test_' + dimension + '_bert_base.csv'
model = gensim.models.fasttext.load_facebook_vectors(cap_path)
normalizer = Normalizer(lang='de', norm_puncts=True)
tokenizer = Tokenizer(lang='de')

def clean_text(row):
    text = row['text']
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)
    text = re.sub(r"http\S+", "", text,re.IGNORECASE)
    text = re.sub(r"RT @\S+", "", text,re.IGNORECASE)
    text = re.sub(r"@\S+", "", text,re.IGNORECASE)
    text = re.sub(emoji.get_emoji_regexp(), "", text)
    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)
    return text

def get_train_test_data():
    data = pd.read_csv('data/polly/polly_by_party_' + dimension + '.csv')
    print(dimension, data.shape)
    del data['Unnamed: 0']
    data_0 = pd.DataFrame()
    data_1 = pd.DataFrame()
    if dimension == afd_green:
        data_0 = data[data['Party']=='Die Grünen']
        data_1 = data[data['Party']=='AfD']
    else: 
        data_0 = data[data['Party']=='Die Linke']
        data_1 = data[data['Party']=='FDP']
        
    if (data_0.shape[0]>data_1.shape[0]):
        data_0 = data_0.sample(data_1.shape[0])
    else:
        data_1 = data_1.sample(data_0.shape[0])
        
    train= pd.DataFrame()
    train = train.append(data_0.sample(frac=0.9))
    test = pd.DataFrame()
    test= test.append(data_0.drop(train.index))
    train = train.append(data_1.sample(frac=0.9))
    if dimension == afd_green:
        test = test.append(data_1.drop(train[train['Party']=='AfD'].index))
    else:
        test = test.append(data_1.drop(train[train['Party']=='FDP'].index))
    train = train.dropna()
    test = test.dropna()
    train.to_csv('data/polly/polly_train_' + dimension + '_fast_text.csv')
    test.to_csv('data/polly/polly_test_' + dimension + '_fast_text.csv')
    
def sent_vectorizer(sent, model):
    print(sent, len(sent))
    sent = normalizer.normalize(sent)
    sent = tokenizer.tokenize(sent)
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    x = np.asarray(sent_vec) / numw
    if (x.shape[0] !=300):
        print(x.shape, sent)
    return x

get_train_test_data()
# train_data = pd.read_csv(path_to_train_data)
# train_data['text'] = train_data.apply(clean_text, axis=1)
# train_data['text'].replace('', np.nan, inplace=True)
# train_data['text'].replace(' ', np.nan, inplace=True)
# train_data['text'].replace('  ', np.nan, inplace=True)
# train_data['text'].replace('   ', np.nan, inplace=True)
# train_data['text'].replace('    ', np.nan, inplace=True)
# train_data = train_data.dropna() 

# test_data = pd.read_csv(path_to_test_data)
# test_data['text'] = test_data.apply(clean_text, axis=1)
# test_data['text'].replace('', np.nan, inplace=True)
# test_data['text'].replace(' ', np.nan, inplace=True)
# test_data['text'].replace('  ', np.nan, inplace=True)
# test_data['text'].replace('    ', np.nan, inplace=True)
# test_data['text'].replace('     ', np.nan, inplace=True)
# test_data = test_data.dropna()

# V=[]
# for sentence in train_data['text'].tolist():
#     V.append(sent_vectorizer(sentence, model))   

# X=[]
# for sentence in test_data['text'].tolist():
#     X.append(sent_vectorizer(sentence, model)) 
     
# X_train = V
# X_test = X
# Y_train = train_data['labels'].tolist()
# Y_test =  test_data['labels'].tolist()

# classifiers = [
#     LogisticRegression(solver="sag", random_state=1),
#     LinearSVC(random_state=1),
#     RandomForestClassifier(random_state=1),
#     XGBClassifier(random_state=1),
#     MLPClassifier(
#         solver="adam",
#         hidden_layer_sizes=(12, 12, 12),
#         activation="relu",
#         early_stopping=True,
#         max_iter=400
#     ),
# ]
# # get names of the objects in list 
# names = [re.match(r"[^\(]+", name.__str__())[0] for name in classifiers]
# print(f"Classifiers to test: {names}")

# results = {}
# for name, clf in zip(names, classifiers):
#     print(f"Training classifier: {name}")
#     clf.fit(X_train, Y_train)
#     prediction = clf.predict(X_test)
#     report = sklearn.metrics.classification_report(Y_test, prediction)
#     results[name] = report
    
# # Prediction results
# for k, v in results.items():
#     print(f"Results for {k}:")
#     print(f"{v}\n")
    




