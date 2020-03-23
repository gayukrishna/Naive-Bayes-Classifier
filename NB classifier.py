# Created by: Gayathri Krishnamoorthy
# Updated: 03-23-2020

# Here, a naive bayesian classifier is implemented with laplace smoothing for fortune cookie data classification.
# It is coded in python version 3.6.

import numpy as np
import csv
import os
from math import floor
import matplotlib.pyplot as plt
from numpy import linalg
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from collections import defaultdict
import re

## get fortune cookie data for removing stop words

x_train = open("./fortunecookiedata/traindata.txt","r")
x_test = open("./fortunecookiedata/testdata.txt","r") 
x_stop = open("./fortunecookiedata/stoplist.txt","r")

## remove the stop words from training set and create new_fortunes_train
stop_h = x_stop
fortune_h = x_train

fortune_list = []
stoplist = stop_h.read().splitlines()
for line in fortune_h:
    fortune_list.append(line.split())

fout = open('./fortunecookiedata/new_fortunes_train.txt','w')

for line in fortune_list:
    print(line)
    for word in stoplist:
        if word in line:
            line.remove(word)
            print(line)
    _line = ' '.join(word for word in line)
    fout.write(_line+'\n')

## remove the stop words from test set and create new_fortunes_test
stop_h = x_stop
fortune_h = x_test

fortune_list = []
stoplist = stop_h.read().splitlines()
for line in fortune_h:
    fortune_list.append(line.split())

fout = open('./fortunecookiedata/new_fortunes_test.txt','w')

for line in fortune_list:
    print(line)
    for word in stoplist:
        if word in line:
            line.remove(word)
            print(line)
    _line = ' '.join(word for word in line)
    fout.write(_line+'\n')


## the data set is cleaned (multiple spaces are replaced with single spaces, all special characters are removed, 
## and the data is converted to lower case)

def clean_data(fortunes):
    
    cleaned_set=re.sub('[^a-z\s]+',' ',fortunes,flags=re.IGNORECASE) # anything other than characters are replaced with ''.
    cleaned_set=re.sub('(\s+)',' ',cleaned_set) # multiple spaces replaced with single space
    cleaned_set=cleaned_set.lower() # changing all the alphabets to lower case    
    return cleaned_set

class NaiveBayeswithLaplace:
    
    def __init__(self,number_of_classes):
        
        self.binaryclass=number_of_classes         

    def bowvocab(self,data,index): ## forming a bag of words dictionary 

        if isinstance(data,np.ndarray):             
            data = data[0]
     
        for word in data.split():           
            self.vocab[index][word]+=1 
            
    def trainfortunes(self,data,labels):
    
        self.data=data
        self.labels=labels
        self.vocab=np.array([defaultdict(lambda:0) for index in range(self.binaryclass.shape[0])])     
 
        ## creating feature set for each fortune in the taining set
    
        for index,rand in enumerate(self.binaryclass):
          
            fortunelist=self.data[self.labels==rand] 
            
            cleaned_set=[clean_data(fortunes) for fortunes in fortunelist]
            
            cleaned_set=pd.DataFrame(data=cleaned_set)

            np.apply_along_axis(self.bowvocab,1,cleaned_set,index) ## categorizing for each fortune
            
                
        ###################################################################################################
        ###################################################################################################
        
        fortuneset=[]
        NBprob=np.empty(self.binaryclass.shape[0])
        countprob=np.empty(self.binaryclass.shape[0])
        
        for index,rand in enumerate(self.binaryclass):

            NBprob[index]=np.sum(self.labels==rand)/float(self.labels.shape[0]) # calculating prob for each fortune class
                       
            count=list(self.vocab[index].values())
            countprob[index]=np.sum(np.array(list(self.vocab[index].values())))+1 ## getting all the words in the fortune list
                               
            fortuneset+=self.vocab[index].keys()                                                 
       
        self.bowdict=np.unique(np.array(fortuneset))
        self.bowdict_length=self.bowdict.shape[0]
                                                                       
        NBden=np.array([countprob[index]+self.bowdict_length+1 for index,rand in enumerate(self.binaryclass)])                                                                          
        self.alltuple=[(self.vocab[index],NBprob[index],NBden[index]) for index,rand in enumerate(self.binaryclass)]                               
        self.alltuple=np.array(self.alltuple)                                 
                                              
                                              
    def GetProb(self,sample):                         
                            
        LHprob=np.zeros(self.binaryclass.shape[0]) 
        
        for index,rand in enumerate(self.binaryclass):                              
            for sample_token in sample.split():                            
                sample_counts=self.alltuple[index][0].get(sample_token,0)+1                
                sample_prob=sample_counts/float(self.alltuple[index][2])
                LHprob[index]+=np.log(sample_prob)

        posteriorprob=np.empty(self.binaryclass.shape[0])
        
        for index,rand in enumerate(self.binaryclass):
            posteriorprob[index]=LHprob[index]+np.log(self.alltuple[index][1])                               
        return posteriorprob

    def trainprob(self,train_data):

        newpred=[] 
        for fortune in train_data:                                  
            clean_set=clean_data(fortune)                                  
            postprob=self.GetProb(clean_set) 
            newpred.append(self.binaryclass[np.argmax(postprob)])

        return np.array(newpred)
    
   
    def testfortunes(self,test_data):
        
        newpred=[] 
        for fortune in test_data:                                  
            clean_set=clean_data(fortune)                                  
            postprob=self.GetProb(clean_set) 
            newpred.append(self.binaryclass[np.argmax(postprob)])
                
        return np.array(newpred)

## getting data input for training and testting set
x_train = open("./fortunecookiedata/new_fortunes_train.txt").read().splitlines()
y_train = open("./fortunecookiedata/trainlabels.txt").read().splitlines()
x_test = open("./fortunecookiedata/new_fortunes_test.txt").read().splitlines()
y_test = open("./fortunecookiedata/testlabels.txt").read().splitlines()

train_data = np.asarray(x_train)
train_labels = np.asarray(y_train)
test_data = np.asarray(x_test)
test_labels = np.asarray(y_test)

# print(type(train_data))
# print(type(train_labels))
# print(type(test_data))
# print(type(test_labels))

## training using train data

NBLpointer=NaiveBayeswithLaplace(np.unique(train_labels))
print ("---------------- Training the fortune set --------------------")
NBLpointer.trainfortunes(train_data,train_labels) 

## get training accuracy 
train_pred=NBLpointer.trainprob(train_data)
training_acc=np.sum(train_pred==train_labels)/float(train_labels.shape[0]) 
print ("Fortunes in training set: ",train_data.shape[0])
print ("Training accuracy: ",training_acc*100,"%")

print ("---------------- Testing the fortune set --------------------")
## test using test set
prediction=NBLpointer.testfortunes(test_data) 

## getting accuracy based on test set
test_acc=np.sum(prediction==test_labels)/float(test_labels.shape[0]) 
print ("Fortunes in testing set: ",test_labels.shape[0])
print ("Testing accuracy: ",test_acc*100,"%")
