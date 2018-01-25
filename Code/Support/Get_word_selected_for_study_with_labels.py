import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys
from sklearn.externals import joblib
from nltk.corpus import wordnet


# We will use this program to select 300 words common in word vectors and CNN and associate them with their labels

selected_words  =joblib.load('ImageNet_words_selected.pkl')


label_dict  =joblib.load('ImageNet_labels.pkl')
print len(selected_words)
selected_words = list(set(selected_words))
selected_words.sort()
print len(selected_words)

label_dict_of_selected ={}
for word in selected_words:
	label_dict_of_selected[word] = label_dict[word]


print len(label_dict_of_selected)

joblib.dump(label_dict_of_selected,"selected_with_labels.pkl")