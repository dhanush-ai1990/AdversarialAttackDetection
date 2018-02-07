#This program will load the word vectors and check if the similarity of word vectors has an impact on adversablity of the 
#network
import gensim
from sklearn.externals import joblib
#from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from random import shuffle
import random
mapping =joblib.load("/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wnid_label_name_with_path.pkl")

#lets load the word vectors

vectors =joblib.load("/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wordVectors.pkl")

#wv=open("/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wv.txt",'w')
#wv.write('392 300'+'\n')
vec= vectors.keys()
vec.sort()
"""
for v in vec:
	string= v +' '+ " ".join([str(i) for i in vectors[v]])+'\n'
	wv.write(string)

wv.close()
"""

#/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/Skip_gram_corrected.txt
word_vectors = Word2Vec.load_word2vec_format('/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wv.txt', binary=False)
#word_vectors = Word2Vec.load_word2vec_format('/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/Skip_gram_corrected.txt', binary=False)
#print word_vectors.most_similar('television',topn=392)

#Select 100 random word vectors
#random.seed(4674488)
#shuffle(vec)
#selected_100_vectors=vec[0:100]

#selected_100_vectors.sort()

#print selected_100_vectors
#out=word_vectors.most_similar('abacus',topn=392)
#temp=[out[0],out[1],out[60],out[120],out[-5],out[-1]]
#print temp

#Get 5 classes to generare adversarial examples, Ideally two closest, two in the middle and two the farthest in cosine
#Distance
# We use gensim to accomplish this.
selected_100_vectors_trails=[]
selected_100_vectors=[]
#for word in selected_100_vectors:
for word in vec:
	out=word_vectors.most_similar(word,topn=392)
	print len(out)
	temp=[out[0],out[80],out[160],out[240],out[320],out[-1]]
	if out[0][1] > 0.547:
		selected_100_vectors.append(word)
		selected_100_vectors_trails.append(temp)
print (selected_100_vectors)
print len(selected_100_vectors_trails)
print len(selected_100_vectors)
#we have selected 6 words per each class for these 100 words.
#Dump and save these as pickles.

#joblib.dump([selected_100_vectors,selected_100_vectors_trails],"/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/adversarialstudy_selected100_with6classes.pkl")




