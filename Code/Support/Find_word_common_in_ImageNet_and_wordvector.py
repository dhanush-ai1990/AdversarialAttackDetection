# This program extracts words in word Vectors and Imagnet for Adverserial Study.

import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys


DICTIONARY ="/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/Dictionary/dictionary_org.txt"
IMG_PATH ='/Users/Dhanush/Desktop/Projects/BrainBench-CNNVisualizations/deep-learning-models/DataScienceImages/'
layer_name = 'flatten_1'
layer_name ='conv1'
word_vec_in ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/"
size_words =60
from sklearn.externals import joblib
DICTIONARY  =joblib.load('ImageNet_words.pkl')
print len(DICTIONARY)


def get_matrix_and_mask(vector_file):
	unavailable = []	# list of indexes of word in brain data that did not appear in the input
	word_vector = []	# input word vector

	# dic for dictionary
	dictionary = {}
	for word in DICTIONARY:
		dictionary[word] = 0
	# dic for input vectors
	input_words = {}
	# filter out words from the input that is not in the dictionary
	count = 0
	added_word = {}
	for index, line in enumerate(vector_file):
		tokens = line.strip().split()
		word = tokens.pop(0)
		count+=1;
		word = word.lower()										
		if word in dictionary: 
			if word not in added_word:						
				input_words[word] = (map(float, tokens))
				added_word[word] = 0
	#print len(dictionary)
	# find words that is in dictionary but not in the input, record their indexs for making a mask
	for word in DICTIONARY:
		if word not in input_words: 
			unavailable.append(word) 
	keylist = input_words.keys()
	keylist.sort()
	for key in keylist:
	    word_vector.append(input_words[key])
	#print len(keylist)
	    # print "%s: %s" % (key, input_words[key])
	# print word_vector

	word_vector = np.array(word_vector)


	length = len(word_vector)
	return added_word

def main():
	

	input_file_list =['Global_context.txt','RNN.txt','Cross_lingual.txt','glove.6B.300d.txt','Skip_gram_corrected.txt','Non-Distributional.txt']
	#input_file_list = ['Skip_italian.txt']
	available_words =[]
	for z in range(6):
		
		input_vec = word_vec_in +str(input_file_list[z])
		input_file= open(input_vec,'r')
		print str(input_file_list[z])
		words = get_matrix_and_mask(input_file)
		available_words.append(words)
		print len(words)

	GC = set(available_words[0])
	RNN=set(available_words[1])
	CL=set(available_words[2])
	GLO=set(available_words[3])
	SG=set(available_words[4])
	ND = set(available_words[5])

	common = GC & RNN & CL & GLO & SG & ND
	print (common)
	print len(common)
	common_selected = list(common)
	common_selected.sort()
	joblib.dump(common_selected,"ImageNet_words_selected.pkl")

		
		
if __name__ == "__main__":
    main()

