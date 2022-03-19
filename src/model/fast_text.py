from gensim.test.utils import datapath
import gensim
import numpy as np
import pandas as pd
from icu_tokenizer import Tokenizer
from icu_tokenizer import Normalizer
from sklearn.neural_network import MLPClassifier

path_to_train_data = 'data/polly/polly_train_fdp_linke_bert_base.csv'
path_to_test_data = 'data/polly/polly_test_fdp_linke_bert_base.csv'
path_to_save_model = 'models/fasttext-mlp-fdp-linke.pt'

cap_path = datapath("/netscratch/aishwarya/political-bias-classification/data/fasttext/cc.de.300.bin")
model = gensim.models.fasttext.load_facebook_vectors(cap_path)
normalizer = Normalizer(lang='de', norm_puncts=True)
tokenizer = Tokenizer(lang='de')
text = "Heute stehen sie leider Seit an Seit mit denen , die #KrimAnnexion verharmlosen. Motto: d Wirtschaft dienen, u zwar d deutschen. Liberal???"
text = normalizer.normalize(text)
print(tokenizer.tokenize(text))

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
    sent_vec =np.array([])
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.append(sent_vec, model[w])
            numw+=1
        except:
            pass
    
    return np.mean(sent_vec)

def meanEmbeddingVectorizer(sentences, model):
    return np.array([
            np.mean([model[w] for w in sent])
            for sent in sentences
        ])

def vectorize(sent, model):
    sent = normalizer.normalize(sent)
    sent = tokenizer.tokenize(sent)
    sent_vec =[]
    return np.mean([model[w] for w in sent]) 

train_data = pd.read_csv(path_to_train_data)
test_data = pd.read_csv(path_to_test_data)
get_train_test_data()
train_vectors = np.array([])
for tweet in train_data['text'].tolist():
     train_vectors = np.append(train_vectors, vectorize(tweet, model))   
test_vectors = []
for tweet in test_data['text'].tolist():
    test_vectors.append(vectorize(tweet, model))  

# train_vectors = []
# for tweet in train_data['text'].tolist():
#     tweet = normalizer.normalize(tweet)
#     tweet = tokenizer.tokenize(tweet)
#     train_vectors.append(tweet)   
# test_vectors = []
# for tweet in test_data['text'].tolist():
#     tweet = normalizer.normalize(tweet)
#     tweet = tokenizer.tokenize(tweet)
#     test_vectors.append(tweet) 
    
# X_train = meanEmbeddingVectorizer(train_vectors,model)
# print("##########################")
# print(X_train)
# print("##########################")
# X_test = meanEmbeddingVectorizer(test_vectors,model)
# Y_train = train_data['labels'].tolist()
# Y_test =  test_data['labels'].tolist()

    
X_train = train_vectors
X_test = test_vectors
Y_train = train_data['labels'].tolist()
Y_test =  test_data['labels'].tolist()
     
classifier = MLPClassifier(alpha = 0.7, max_iter=400) 
classifier.fit(X_train, Y_train)
 
df_results = pd.DataFrame(data=np.zeros(shape=(1,3)), columns = ['classifier', 'train_score', 'test_score'] )
train_score = classifier.score(X_train, Y_train)
test_score = classifier.score(X_test, Y_test)
 
print(classifier.predict_proba(X_test))
print(df_results)