import os, sys
from model2 import Model2 as Model
from DataParser import DataParser
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np


def thresholdTuning(tr,pr):
    pre = set(pr)
    pre=set([round(elem,4) for elem in pre])
    bestF=0
    bestThre=0
    pr=np.array(pr)
    for thre in pre:
        scr=f1_score(tr,pr>=thre)
        if scr>bestF:
            bestF=scr
            bestThre=thre
    return bestF,bestThre

 
def ComputeFscore(modelfile,testfile,outputfile):

    CURRENT_DIR = os.path.dirname(os.path.abspath("./WikiCategoryLabelling/"))
    sys.path.append(os.path.dirname(CURRENT_DIR+"/WikiCategoryLabelling/"))

    maxParagraphLength=250
    maxParagraphs=10
    labels=1000
    vocabularySize=150000
    model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)

    testing = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
    testing.getDataFromfile(testfile)

    model.load(modelfile)

    print("loading done")

    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch()
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])

    labelsCount={}
    ConfusionMa={}
    fScr={}

    thres=0.5
    valid=int(len(truePre)*0.35)
    labelsCount={}
    ConfusionMa={}
    fScr={}
    thresLab={}
    for la in range(1000):
        if la%25==0:
            print("Currnet label",la)
        t=[]
        p=[]
        for i in range(valid):
            t.append(truePre[i][la])
            p.append(pred[i][la])
        bestF,bestThre=thresholdTuning(t,p)
    
        t=[]
        p=[]
        for i in range(valid,len(truePre)):
            t.append(truePre[i][la])
            p.append(pred[i][la])
    
        p=np.array(p)
        fScr[la]=f1_score(t,p>=bestThre)
        ConfusionMa[la]= confusion_matrix(t,p>bestThre)
        thresLab[la]=bestThre
    
    f=open(outputfile,"w")
    for i in range(1000):
        inp=str(i)+","+str(thresLab[i])+","+str(fScr[i])+"\n"
        f.write(inp)
    f.close()


if __name__ == '__main__':
    modelfile = sys.argv[1]
    testfile = sys.argv[2]
    outputfile = sys.argv[3]

    ComputeFscore(modelfile,testfile,outputfile)

