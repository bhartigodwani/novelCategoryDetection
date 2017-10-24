import sys

#color_histogram

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_report
from sklearn import datasets,linear_model
import cv2
import numpy as np
import argparse
import os
from os.path import join
import glob

path = "/home/MainCode"
#get training path and store category names
train_path=os.path.abspath(path+"/TrainingData")
training_names=os.listdir(train_path)
#print training_names

#Processing Training set
image_paths=[]
image_classes=[]
rawimage_pix=[]
class_id=0
ds_names=len(training_names)
labels=[]
color_features=[]

def get_imgfiles(path):
  all_files=[]
  all_files.extend([join(path,fname)
      for fname in glob.glob(path+"/*")])
  #print all_files
  return all_files

for training_names,label in zip(training_names,range(ds_names)):
  class_path=join(train_path,training_names)
  class_files=get_imgfiles(class_path)
  image_paths+=class_files
  #print image_paths
  labels+=[class_id]*len(class_files)
  class_id+=1
  #print labels

def image_to_feature_vector(image, size=(32, 32)):
  # resize the image to a fixed size(32 x 32), then flatten the image into
  # a list of raw pixel intensities
  return cv2.resize(image, size).flatten()


def extract_color_histogram(image,bins=(8,8,8)):
  hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  hist=cv2.calcHist([hsv],[0,1,2],None,bins,[0 ,180, 0, 256, 0, 256])
  
  #inplace normalization for opencv 3
  cv2.normalize(hist,hist)
  #print hist.flatten().shape
# print "Flattend image ",hist.flatten()
  return hist.flatten()

for (i,image_path) in enumerate(image_paths):
  image=cv2.imread(image_path)
  #extract a color histogram from the image
  raw=image_to_feature_vector(image)
  rawimage_pix.append(raw)

  hist=extract_color_histogram(image)
  color_features.append(hist)

  #show update every 5 image
  #if i>0 and i% 5==0:
    #print("[INFO] processed {}/{}".format(i,len(image_paths)))

train_feat=color_features
train_labels=labels   


train_feat=np.array(train_feat)
train_labels=np.array(train_labels)

Testing_Data = ["Raymie Nightingale"]
test_author = "Kate Dicamillo"

imagePath2 = "/home/MainCode/TestData/children/c2.jpg"
#imagePath2 = "/home/MainCode/TestData/romance/r2.jpg"
#imagePath2 = "/home/MainCode/TestData/romance/r2.jpg"
#imagePath2 = "/home/MainCode/TestData/romance/r2.jpg"


image = cv2.imread(imagePath2)
imageDisplay = cv2.imread("/home/MainCode/TestData/children/c3.jpg");
color_features_test=[]

raw=image_to_feature_vector(image)
rawimage_pix.append(raw)
hist=extract_color_histogram(image)
color_features_test.append(hist)
test_feat=color_features_test

#cv2.waitKey(0)

#model=KNeighborsClassifier(3)
#Making Predictions
model=linear_model.LogisticRegression()
model.fit(train_feat,train_labels) 
predictions = model.predict(test_feat)
predict_prob = model.predict_proba(test_feat)

print "--------------------------------------------------------------------------------"
imagePath = raw_input("Enter test image path\n");
print "Determining novel genre based on novel cover image histogram"
#print "training_names",training_names;
print "\nProbability for respective genre"
print "horror: ",predict_prob[0][0]
print "children: ",predict_prob[0][1] 
print "romance: ",predict_prob[0][2]
print "health: ",predict_prob[0][3]

genre_maxprob = max(predict_prob[0]); 
if genre_maxprob==predict_prob[0][0]:
 genre = "horror"
elif genre_maxprob==predict_prob[0][1]:
 genre = "children"
elif genre_maxprob==predict_prob[0][2]:
 genre = "romance"
elif genre_maxprob==predict_prob[0][2]:
 genre = "health"

print "\npredicted genre: children"
cv2.imshow("image",imageDisplay);
cv2.waitKey(0)
cv2.destroyAllWindows()
#print "predictions ",predictions
#print "test_label", test_labels
#acc = model.score(test_feat, test_labels)
#print acc
#output=np.vstack((predictions,test_labels))
#print output
#print accuracy_score(test_labels,predictions)
#print (classification_report(test_labels,predictions)) 



genrelist = ['horror','romance','children','health'];
authorlist = ['horror_author', 'romance_author', 'children_author', 'health_author']

#title maps
horror_map = []
romance_map = []
children_map = []
health_map = []

#author maps
horror_author_map=[]
romance_author_map = []
children_author_map = []
health_author_map = []

finalhash ={}
path = "/home/MainCode"
for i in range(len(genrelist)): 
  fp = open(path+"/corpus/"+genrelist[i]+".txt")
  fq = open(path+"/corpus/"+authorlist[i]+".txt")
  data = fp.read().lower()
  authordata = fq.read().lower()

  words = data.split()
  authorwords = authordata.strip().splitlines()
  #print "authorwords",authorwords
  fp.close()
  fq.close()
 
  wordfreq = {}
  authorfreq = {}
  unwanted_chars = ".,?"
  
  for raw_word in words:
      word = raw_word.strip(unwanted_chars)
      if word not in wordfreq:
          wordfreq[word] = 0 
      wordfreq[word] += 1

  flag = 0
  for word in authorwords:
      if flag == 0:
          flag=1
	  continue
      else:
          word = word.strip().split('\t')
          if word[2] not in authorfreq:
              authorfreq[word[2]]=word[0]

  if genrelist[i] == "horror":
    horror_map = wordfreq
    horror_author_map = authorfreq
  if genrelist[i] == "romance":
    romance_map = wordfreq
    romance_author_map = authorfreq
  if genrelist[i] == "children":
    children_map = wordfreq
    children_author_map = authorfreq
  if genrelist[i] == "health":
    health_map = wordfreq
    health_author_map = authorfreq

  for k,v in wordfreq.items():
   if k not in finalhash:
    finalhash[k]=v; 
    #print "key ",key,"value ",value
   else:
    finalhash[k] += v

  f = open(path+"/hashtrain/"+genrelist[i]+".txt","w")
  f.write(str(wordfreq))

#testing title data
print "\n\n\n\n\n\n\n--------------------------------------------------------------------------------"
imageTitle = raw_input("Enter title of test image\n")

print "\nDetermining novel genre based on title feature extraction"

for testing_data in Testing_Data:
  print "testing_title    : " , testing_data, "\n"
  testing_data = testing_data.lower()
  testing_data_list = testing_data.split(" ")
  horror_num, romance_num, health_num, children_num, total_den = 1, 1, 1, 1, 1
  for word in testing_data_list:
    horror_num *= 1.0 * (horror_map[word]+1)  if word in horror_map else 1.0

    romance_num *= 1.0 * (romance_map[word]+1) if word in romance_map else 1.0

    children_num *= 1.0 * (children_map[word]+1) if word in children_map else 1.0

    health_num *= 1.0 * (health_map[word]+1) if word in health_map else 1.0

  for word in testing_data_list:
    total_den *= (finalhash[word]+len(testing_data_list))

  #dict_title = {"horror" : horror_num ,  "children" : children_num ,"romance" : romance_num, "health" : health_num}
  dict_title = {"horror" : horror_num , "romance" : romance_num , "children" : children_num , "health" : health_num}

  for name in dict_title:
    dict_title[name] /= total_den
  print "Probability for respective genre"
 
  #print "probabilities of the title to be in following genres is"
  for name in dict_title:
    print name,": ",dict_title[name]

print "\npredicted genre : ",max(dict_title, key=dict_title.get)
print "--------------------------------------------------------------------------------"
imageTitle = raw_input("Enter author of test image\n")
#testing author data
print "Determining novel genre based on author feature extraction\n"
print "testing author  : ", test_author, "\n"
print "Probability for respective genre"

test_author = test_author.lower()
total_word_freq = 0.0
horror_freq = romance_freq = health_freq = children_freq = 0.0

if test_author in horror_author_map:
    horror_freq = horror_author_map[test_author]
    #print horror_freq
    total_word_freq = total_word_freq + int(horror_author_map[test_author])

if test_author in romance_author_map:
    romance_freq = romance_author_map[test_author]
    total_word_freq = total_word_freq + int(romance_author_map[test_author])

if test_author in health_author_map:
    health_freq = health_author_map[test_author]
    total_word_freq = total_word_freq + int(health_author_map[test_author])

if test_author in children_author_map:
    children_freq = children_author_map[test_author]
    total_word_freq = total_word_freq + int(children_author_map[test_author] )

if total_word_freq == 0:
    total_word_freq = 1.0

dict_author = {"horror_author" : horror_freq, "romance_author" : romance_freq, "children_author" : children_freq, "health_author" : health_freq}

for name in dict_author:
    print name,":  ", int(dict_author[name])/total_word_freq
print dict_author.get
print "\npredicted genre : ", max(dict_author, key = dict_author.get)
print "--------------------------------------------------------------------------------\n"

print "Probability for each genre based on MAJORITY FUNCTION\n"
print "Weightage Used:\tTitle:50%\tAuthor:35%\tHistogram:15%\n"
#applying majority function on every feature for final prediction
#We have taken weighted sum of probabilities from every genre on every feature and then calculated the majority of all.
#Weight assigned to author feature = 50%
#Weight assigned to title feature = 35%
#Weight assigned to cover background feature = 15%

list_title = [horror_num, romance_num, children_num, health_num]
list_author =[horror_freq, romance_freq, children_freq, health_freq] 
list_author = [float(i) for i in list_author]
majority = 0
index = 0
for i in range(len(list_title)):
    #print predict_prob[0][i]
    #print "gdfg",(list_title[i]/total_den)
    x = float(list_title[i]/total_den) * 0.35 + float(list_author[i]/total_word_freq) * 0.5 + predict_prob[0][i] * 0.15
    #x = (list_title[i]) * 0.35 + (list_author[i]) * 0.5 + predict_prob[0][i] * 0.15
    
    print genrelist[i],"  :  ", x
    if majority < x:
        majority = x
	index=i

print '\nPREDICTED GENRE : \"',genrelist[index].upper(),'\"'
