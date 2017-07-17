import os, sys
from model2_siml import Model2_siml as Model
from DataParser_siml import DataParser_siml as DataParser
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import math


def ComputePrecisionK(modelfile,testfile,nhead,K_list):

    CURRENT_DIR = os.path.dirname(os.path.abspath("./WikiCategoryLabelling/"))
    sys.path.append(os.path.dirname(CURRENT_DIR+"/WikiCategoryLabelling/"))

    maxParagraphLength=2500
    maxParagraphs=1
    labels=1001
    vocabularySize=76391
    model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)

    testing = DataParser(maxParagraphLength,labels,vocabularySize)
    testing.getDataFromfile(testfile)
    print("data loading done")
    print("no of test examples: " + str(testing.totalPages))

    model.load(modelfile)

    print("model loading done")

    batchSize = 10

    testing.restore()
    truePre=[]
    pred=[]
    for i in range(math.ceil(testing.totalPages/batchSize)):
        if i < testing.totalPages/batchSize:
            data=testing.nextBatch(batchSize)
        else:
            data=testing.nextBatch(testing.totalPages%batchSize)
        truePre.extend(data[0])
        pre=model.predict(data)
        pred.extend(pre[0].tolist())

    avgHeadPrecK = [0]*len(K_list)
    avgTailPrecK = [0]*len(K_list)
    avgPrecK = [0]*len(K_list)
    for i,p in enumerate(pred):
        sortedHeadL = sorted(range(len(p[:nhead])), key=p[:nhead].__getitem__, reverse=True)
        sortedTailL = sorted(range(len(p[nhead:])), key=p[nhead:].__getitem__, reverse=True)
        sortedL = sorted(range(len(p)), key=p.__getitem__, reverse=True)
        for k,K in enumerate(K_list):
            headLabelK = sortedHeadL[:K]
            tailLabelK = sortedTailL[:K]
            labelK = sortedL[:K]
            headPrecK = 0
            tailPrecK = 0
            precK = 0
            for l in headLabelK:
                if truePre[i][l] == 1 and l < nhead:
                    headPrecK += 1
            for l in tailLabelK:
                if truePre[i][l] == 1 and l >= nhead:
                    tailPrecK += 1
            for l in labelK:
                if truePre[i][l] == 1:
                    precK += 1
            avgHeadPrecK[k] += headPrecK/float(K)
            avgTailPrecK[k] += tailPrecK/float(K)
            avgPrecK[k] += precK/float(K)
    avgHeadPrecK = [float(a)/len(pred) for a in avgHeadPrecK]
    avgTailPrecK = [float(a)/len(pred) for a in avgTailPrecK]
    avgPrecK = [float(a)/len(pred) for a in avgPrecK]
           
    print("Precisions for head labels")
    for p in avgHeadPrecK:
        print(str(p))
    print("\nPrecisions for tail labels")
    for p in avgTailPrecK:
        print(str(p))
    print("\nPrecisions for all labels")
    for p in avgPrecK:
        print(str(p))


if __name__ == '__main__':
    modelfile = sys.argv[1]
    testfile = sys.argv[2]
    nhead = int(sys.argv[3])
    K_list = [1,2,3,4,5,6,7,8,10,15,20]

    ComputePrecisionK(modelfile,testfile,nhead,K_list)

