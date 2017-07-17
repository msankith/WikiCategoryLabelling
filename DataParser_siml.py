
# coding: utf-8

# In[ ]:

import numpy as np
class DataParser_siml:
    def __init__(self,paraLength,labels,vocabSize):
        self.data=[]
        self.paragraphLength=paraLength
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
        maxParagraphs=1
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
            docText = []
            for i in range(instancesCount):
                tempInstance=f.readline().split()
                temp=[int(x) for x in tempInstance if int(x) > 0 and int(x) < self.vocabSize]
                docText.extend(temp)
            if len(docText) > maxWordsInParagraph:
                docText = docText[:maxWordsInParagraph]
            else:
                for i in range(len(docText),maxWordsInParagraph):
                    docText.append(0)
            self.data.append((labelsTemp,docText))
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
        labelBatch=[]
        feaBatch=[]
        for i in range(batchSize):
            if self.counter+1 >=self.totalPages:
                self.counter=0
            labelBatch.append(self.data[self.counter][0])
            feaBatch.append(self.data[self.counter][1])
            self.counter+=1
        return (labelBatch,[feaBatch])
    
    def restore(self):
        self.counter=0

