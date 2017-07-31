import numpy as np
import pickle
class DataParser_wiki10:
    def __init__(self,paraLength,nlabels,vocabSize):
        self.data=[]
        self.paragraphLength=paraLength
        self.nlabels=nlabels
        self.vocabSize=vocabSize

    def getDataFromfile(self,fname):
        features,labels,n_features,n_labels = pickle.load(open(fname),'rb')
        assert len(features) == len(labels)
        assert self.nlabels == n_labels
        self.totalPages = len(features)
        self.features = features
        self.labels = labels
        self.counter = 0
        
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
        featBatch=[]
        featValueBatch=[]
        for i in range(batchSize):
            if self.counter+1 >=self.totalPages:
                self.counter=0
            curLabels = self.labels[self.counter]
            oneHotLabels = [0]*self.nlabels
            for l in curLabels:
                oneHotLabels[l] = 1
            labelBatch.append(oneHotLabels)
            curFeat = self.features[self.counter].keys()
            if len(curFeat) > self.paragraphLength:
                curFeat = curFeat[:self.paragraphLength]
            curFeatValue = [self.features[self.counter][f] for f in curFeat]
            for i in range(len(curFeat),self.paragraphLength):
                curFeat.append(0)
                curFeatValue.append(0)
            featBatch.append(curFeat)
            featValueBatch.append(curFeatValue)
            self.counter+=1
        return (labelBatch,[featBatch],[featValueBatch])
    
    def restore(self):
        self.counter=0

