#This program will select five random images from each class for our study
from sklearn.externals import joblib
from random import shuffle
import os
loc='/Volumes/LLL/ILSVRC2012_ValidationData/'
mapping ='/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wnid_label_name.pkl'
mapping=joblib.load(mapping)

for item in mapping:
	location=loc+mapping[item][1]
	files=os.listdir(location)
	shuffle(files)
	files=files[0:5]
	files=[location+'/'+f for f in files]
	mapping[item].append(files)
	
print mapping

joblib.dump(mapping,"/Users/Dhanush/Desktop/Projects/AdverserialStudy/Data/wnid_label_name_with_path.pkl")