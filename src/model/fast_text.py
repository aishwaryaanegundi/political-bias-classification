# from gensim.models import KeyedVectors
# de_model = KeyedVectors.load_word2vec_format('wiki.de\wiki.de.vec')
# print (model.most_similar('frau'))

# words = []
# for word in model.vocab:
#     words.append(word)
    
# print("Vector components of a word: {}".format(
#     model[words[0]]
# ))
# sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
#             ['this', 'is',  'another', 'machine', 'learning', 'book'],
#             ['one', 'more', 'new', 'book'],
#         ['this', 'is', 'about', 'machine', 'learning', 'post'],
#           ['orange', 'juice', 'is', 'the', 'liquid', 'extract', 'of', 'fruit'],
#           ['orange', 'juice', 'comes', 'in', 'several', 'different', 'varieties'],
#           ['this', 'is', 'the', 'last', 'machine', 'learning', 'book'],
#           ['orange', 'juice', 'comes', 'in', 'several', 'different', 'packages'],
#           ['orange', 'juice', 'is', 'liquid', 'extract', 'from', 'fruit', 'on', 'orange', 'tree']]
          
# import numpy as np
 
# def sent_vectorizer(sent, model):
#     sent_vec =[]
#     numw = 0
#     for w in sent:
#         try:
#             if numw == 0:
#                 sent_vec = model[w]
#             else:
#                 sent_vec = np.add(sent_vec, model[w])
#             numw+=1
#         except:
#             pass
    
#     return np.asarray(sent_vec) / numw
 
# V=[]
# for sentence in sentences:
#     V.append(sent_vectorizer(sentence, model))   
         
# X_train = V[0:6]
# X_test = V[6:9] 
# Y_train = [0, 0, 0, 0, 1,1]
# Y_test =  [0,1,1]    
     
     
# from sklearn.neural_network import MLPClassifier
# classifier = MLPClassifier(alpha = 0.7, max_iter=400) 
# classifier.fit(X_train, Y_train)
 
# df_results = pd.DataFrame(data=np.zeros(shape=(1,3)), columns = ['classifier', 'train_score', 'test_score'] )
# train_score = classifier.score(X_train, Y_train)
# test_score = classifier.score(X_test, Y_test)
 
# print(classifier.predict_proba(X_test))
# print(classifier.predict(X_test))
 
# df_results.loc[1,'classifier'] = "MLP"
# df_results.loc[1,'train_score'] = train_score
# df_results.loc[1,'test_score'] = test_score
# print(df_results)


from gensim.test.utils import datapath
import gensim
cap_path = datapath("/netscratch/aishwarya/political-bias-classification/data/fasttext/cc.de.300.bin")
fbkv = gensim.models.fasttext.load_facebook_vectors(cap_path)
print('frau' in fbkv.key_to_index) # Word is out of vocabulary
oov_vector = fbkv['frau']
print('schon' in fbkv.key_to_index)  # Word is in the vocabulary
iv_vector = fbkv['schon']