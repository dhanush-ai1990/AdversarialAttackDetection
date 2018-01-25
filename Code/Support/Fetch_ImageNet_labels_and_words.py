import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys


DICTIONARY ="/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/Dictionary/dictionary_org.txt"
IMG_PATH ='/Users/Dhanush/Desktop/Projects/BrainBench-CNNVisualizations/deep-learning-models/DataScienceImages/'

import json
word_vec_in ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/"
from sklearn.externals import joblib
#org_interest        =joblib.load(file_loc+'org_interest.pkl')
def get_word_vec_list(vector_file):
	list_words =[]
	added_word =[]
	for index, line in enumerate(vector_file):
		tokens = line.strip().split()
		word = tokens.pop(0)
		word = word.lower()										
		if word not in added_word:						
			added_word.append(word)
			list_words.append(word)


	return list_words



def image_net_labels():
	label_file= open('/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/labels.txt','r')
	label_dict = {}
	list_words =[]

	for line in label_file:
		list1=line.split(",")
		list1 = list1[0:-1]
		label = list1[0]
		list1= list1[1:]
		for word in list1:
			label_dict[word] = label
			list_words.append(word)
	list_words.sort()
	print list_words
	joblib.dump(label_dict, "ImageNet_labels.pkl")
	joblib.dump(list_words,"ImageNet_words.pkl")


def main():
	image_net_labels()
	"""
	input_file_list =['Global_context.txt','Skip_gram_corrected.txt','RNN.txt','Cross_lingual.txt','glove.6B.300d.txt','Non-Distributional.txt']
	#input_file_list = ['Skip_italian.txt']
	for z in range(6):
		
		input_vec = word_vec_in +str(input_file_list[z])
		input_file= open(input_vec,'r')
		list_words = get_word_vec_list(input_file)
		print "Number of words in " + str(input_file_list[z]) + " : " + str(len(list_words))
	"""	
		
if __name__ == "__main__":
    main()