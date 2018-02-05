#Get the Label, Words and their Word net id

from sklearn.externals import joblib
import json
#ImageNet_words_selected.pkl
"""
words_for_test = joblib.load("selected_with_labels.pkl")
#print words_for_test

#Lets create a reverse mapping from label to word
label_word_dict ={}
for word in words_for_test:
	label_word_dict[int(words_for_test[word])] = word

#print label_word_dict
"""
file_url ="/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wnid_mapping.txt"
wnid = open(file_url,"r")
list1 =[]
label_wnid ={}
for line in wnid:
	line=line.split()
	if line[0] == 'end':
		print "Done"
		continue
	elif line[0] == "'uri':":
		continue
	elif line[0] == "'label':":
		continue
	else:
		label_wnid[int(line[0][0:-1])] = 'n'+line[2][1:-4]
		#print int(line[0][0:-1]),'n'+line[2][1:-4]

print label_wnid
"""
mapped_final_dict ={}
for label in label_word_dict:
	mapped_final_dict[label_word_dict[label]]=[label,label_wnid[label]]


print len(mapped_final_dict)
"""
joblib.dump(label_wnid,"wordnetid_labels_mapping.pkl")

