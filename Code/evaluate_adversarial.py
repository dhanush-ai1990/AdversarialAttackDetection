#In this program we will select our 6 trail adversarial classes selected using word2vec. We could generate adversarial 
#test examples for these classes for every 100 images. Then we understand if there is any relationship between cosine distance of
#original class and the adversarial class with respect to the adversariality of the image given as input.

from sklearn.externals import joblib