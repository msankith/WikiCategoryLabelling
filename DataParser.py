
# coding: utf-8

# In[ ]:

import numpy as np
import random
class DataParser:
    def __init__(self,paraLength,maxPara,labels,vocabSize):
        self.data=[]
        self.paragraphLength=paraLength
        self.maxParagraph=maxPara
        self.labels=labels
        self.vocabSize=vocabSize

    def getDataFromfile(self,fname):
        self.counter =0
        self.totalPages=0
        f=open(fname)
        self.data=[]
        totalPages = int(f.readline())
        count=0
        maxWordsInParagraph=self.paragraphLength
        maxParagraphs=self.maxParagraph
        totalLabels=self.labels

        dummyParagraph =[0]*maxWordsInParagraph

        while True:
            pageId= f.readline()
            if not pageId :
                break

            labelCount=int(f.readline())
            labelsTemp=[0]*totalLabels
            for i in range(labelCount):
                tempLab=int(f.readline())
                labelsTemp[tempLab]=1
                assert tempLab<totalLabels
            instancesCount=int(f.readline())
            instancesTemp=[]
            for i in range(instancesCount):
                tempInstance=f.readline().split()
                temp=[int(x) for x in tempInstance if int(x) > 0 and int(x) < self.vocabSize]
                for j in range(len(temp),maxWordsInParagraph):
                    temp.append(0)
                instancesTemp.append(temp[:maxWordsInParagraph])

            for i in range(instancesCount,maxParagraphs):
                instancesTemp.append(dummyParagraph)

            self.data.append((labelsTemp,instancesTemp[:maxParagraphs]))
        self.totalPages = len(self.data)
        f.close()
        
          
        
    def nextBatch(self):
        if self.counter >=self.totalPages:
            self.counter=0
        data= self.data[self.counter]
        self.counter+=1
        return data
    def nextBatch(self,batchSize):
        if self.counter >=self.totalPages:
            self.counter=0
            random.shuffle(self.data)
        labelBatch=[]
        feaBatch=[]
        for itr in range(self.maxParagraph):
            feaBatch.append([])
        for i in range(batchSize):
            if self.counter+1 >=self.totalPages:
                self.counter=0
                random.shuffle(self.data)
            labelBatch.append(self.data[self.counter][0])
            for itr in range(self.maxParagraph):
                feaBatch[itr].append(self.data[self.counter][1][itr])
            self.counter+=1
        return (labelBatch,feaBatch)
    
    def restore(self):
        self.counter=0

