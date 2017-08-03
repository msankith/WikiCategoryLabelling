from DataParser_wiki10 import DataParser_wiki10 as DataParser
from model2_wiki import Model2_wiki as Model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os, sys
import math


def ComputePrecisionK(modelfile,testfile,K_list):

    CURRENT_DIR = os.path.dirname(os.path.abspath("./WikiCategoryLabelling/"))
    sys.path.append(os.path.dirname(CURRENT_DIR+"/WikiCategoryLabelling/"))

    maxParagraphLength=2500
    maxParagraphs=1
    labels=30938
    vocabularySize=101939
    num_filters = 100
    model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize,num_filters)

    testing = DataParser(maxParagraphLength,labels,vocabularySize)
    print(testfile)
    testing.getDataFromfile(testfile)
    print("data loading done")
    print("no of test examples: " + str(testing.totalPages))

    model.load(modelfile)

    print("model loading done")

    batchSize = 1000

    testing.restore()
    truePre=[]
    pred=[]
    for i in range(math.ceil(testing.totalPages/batchSize)):
        if i < testing.totalPages/batchSize-1:
            data=testing.nextBatch(batchSize)
        else:
            data=testing.nextBatch(testing.totalPages%batchSize)
        truePre.extend(data[0])
        pre=model.predict(data)
        pred.extend(pre[0].tolist())

    avgPrecK = [0]*len(K_list)
    for i,p in enumerate(pred):
        sortedL = sorted(range(len(p)), key=p.__getitem__, reverse=True)
        for k,K in enumerate(K_list):
            labelK = sortedL[:K]
            precK = 0
            for l in labelK:
                if truePre[i][l] == 1:
                    precK += 1
            avgPrecK[k] += precK/float(K)
    avgPrecK = [float(a)/len(pred) for a in avgPrecK]
            
    for p in avgPrecK:
        print(str(p))


if __name__ == '__main__':
    modelfile = sys.argv[1]
    testfile = sys.argv[2]
    K_list = [1,2,3,4,5,6,7,8,9,10,15,20]

    ComputePrecisionK(modelfile,testfile,K_list)

