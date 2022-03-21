from gensim.models import KeyedVectors
import pandas as pd
from gensim.test.utils import datapath
import gensim
import numpy as np
from icu_tokenizer import Tokenizer
from icu_tokenizer import Normalizer
import sklearn
from sklearn.neural_network import MLPClassifier

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
print(X_train)
X_test = X
Y_train = train_data['labels'].tolist()
print(Y_train)
Y_test =  test_data['labels'].tolist()

 
     
classifier = MLPClassifier(alpha = 0.7, max_iter=400) 
classifier.fit(X_train, Y_train)
 
df_results = pd.DataFrame(data=np.zeros(shape=(1,3)), columns = ['classifier', 'train_score', 'test_score'] )
train_score = classifier.score(X_train, Y_train)
test_score = classifier.score(X_test, Y_test)
 
print(classifier.predict_proba(X_test))
print(classifier.predict(X_test))
prediction = classifier.predict(X_test)
report = sklearn.metrics.classification_report(Y_test, prediction)
print(report)
df_results.loc[1,'classifier'] = "MLP"
df_results.loc[1,'train_score'] = train_score
df_results.loc[1,'test_score'] = test_score
print(df_results)


