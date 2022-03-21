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

cap_path = datapath("/netscratch/aishwarya/political-bias-classification/data/fasttext/cc.de.300.bin")
path_to_train_data = 'data/polly/polly_train_fdp_linke_bert_base.csv'
path_to_test_data = 'data/polly/polly_test_fdp_linke_bert_base.csv'
path_to_save_model = 'models/fasttext-mlp-fdp-linke.pt'
model = gensim.models.fasttext.load_facebook_vectors(cap_path)
normalizer = Normalizer(lang='de', norm_puncts=True)
tokenizer = Tokenizer(lang='de')

def get_train_test_data():
    data = pd.read_csv('data/polly/polly_by_party_fdp_linke.csv')
    del data['Unnamed: 0']
    data_0 = data[data['Party']=='Die Linke']
    data_1 = data[data['Party']=='FDP']
    data_1 = data_1.sample(data_0.shape[0])
    train= pd.DataFrame()
    train = train.append(data_0.sample(frac=0.9))
    test = pd.DataFrame()
    test= test.append(data_0.drop(train.index))
    train = train.append(data_1.sample(frac=0.9))
    test= test.append(data_1.drop(train[train['Party']=='FDP'].index))
    train = train.dropna()
    test = test.dropna()
    train.to_csv('data/polly/polly_train_fdp_linke_bert_base.csv')
    test.to_csv('data/polly/polly_test_fdp_linke_bert_base.csv')
    
def sent_vectorizer(sent, model):
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
train_data = pd.read_csv(path_to_train_data)
train_data['text'].replace('', np.nan, inplace=True)
train_data['text'].replace(' ', np.nan, inplace=True)
train_data['text'].replace('  ', np.nan, inplace=True)
train_data['text'].replace('   ', np.nan, inplace=True)
train_data = train_data.dropna() 

test_data = pd.read_csv(path_to_test_data)
test_data['text'].replace('', np.nan, inplace=True)
test_data['text'].replace(' ', np.nan, inplace=True)
test_data['text'].replace('  ', np.nan, inplace=True)
train_data['text'].replace('    ', np.nan, inplace=True)
test_data = test_data.dropna()

V=[]
for sentence in train_data['text'].tolist():
    V.append(sent_vectorizer(sentence, model))   

X=[]
for sentence in test_data['text'].tolist():
    X.append(sent_vectorizer(sentence, model)) 
     
X_train = V
X_test = X
Y_train = train_data['labels'].tolist()
Y_test =  test_data['labels'].tolist()

classifiers = [
    LogisticRegression(solver="sag", random_state=1),
    LinearSVC(random_state=1),
    RandomForestClassifier(random_state=1),
    XGBClassifier(random_state=1),
    MLPClassifier(
        solver="adam",
        hidden_layer_sizes=(12, 12, 12),
        activation="relu",
        early_stopping=True,
        max_iter=400
    ),
]
# get names of the objects in list 
names = [re.match(r"[^\(]+", name.__str__())[0] for name in classifiers]
print(f"Classifiers to test: {names}")

results = {}
for name, clf in zip(names, classifiers):
    print(f"Training classifier: {name}")
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    report = sklearn.metrics.classification_report(Y_test, prediction)
    results[name] = report
    
# Prediction results
for k, v in results.items():
    print(f"Results for {k}:")
    print(f"{v}\n")
    




