from sklearn.externals import joblib


selected_classes=joblib.load("/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/selected_with_labels.pkl")
wordnet_mapping=joblib.load("/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wordnetid_labels_mapping.pkl")
word_vec_in ="/Users/Dhanush/Desktop/Projects/Brain_Bench/Word_Vectors/Skip_gram_corrected.txt"
wnid_label_name={}

for item in selected_classes:
	class_name=item
	class_no=int(selected_classes[class_name])
	wnid=wordnet_mapping[class_no]
	wnid_label_name[class_name]=[class_no,wnid]

print len(wnid_label_name)


joblib.dump(wnid_label_name,"/Users/Dhanush/Desktop/wnid_label_name.pkl")

#Select only the required wordvectors and dump them as pickle for faster processing


input_file= open(word_vec_in,'r')
dictionary = {}
labels=wnid_label_name.keys()
labels.sort()
print len(labels)
for line in labels:
	dictionary[line] = 0

count = 0
added_word = {}
input_words = {}
for index, line in enumerate(input_file):
	tokens = line.strip().split()
	word = tokens.pop(0)
	count+=1;
	word = word.lower()										
	if word in dictionary: 
		if word not in added_word:						
			input_words[word] = (map(float, tokens))
			added_word[word] = 0

print len(input_words)
joblib.dump(input_words,"/Users/Dhanush/Desktop/wordVectors.pkl")

