import numpy as np
import json
import sys
import random

def SampleData(infile, outfile, sampleSize):
    count = 0
    f=open(infile)
    totalPages = int(f.readline())
    assert totalPages > sampleSize
    pages = []
    for p in range(totalPages):
        pageDict = {}
        pageId= f.readline().strip()
        if not pageId :
            break
        pageDict['pageId']=pageId
        labelCount=int(f.readline())
        labels = []
        for i in range(labelCount):
            tempLab=int(f.readline())
            labels.append(tempLab)
        pageDict['labels']=labels
        instancesCount=int(f.readline())
        instances = []
        for i in range(instancesCount):
            tempInstance=f.readline().strip()
            instances.append(tempInstance)
        pageDict['instances'] = instances
        pages.append(pageDict)
    f.close()
    
    pages = random.sample(pages,len(pages))
    sampledPages = pages[:sampleSize]
    fout = open(outfile,'w')
    fout.write(str(sampleSize)+'\n')
    for p in sampledPages:
        fout.write(p['pageId']+'\n')
        fout.write(str(len(p['labels']))+'\n')
        for l in p['labels']:
            fout.write(str(l)+'\n')
        fout.write(str(len(p['instances']))+'\n')
        for i in p['instances']:
            fout.write(i+'\n')
    fout.close()
        
if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ts = int(sys.argv[3])
    SampleData(infile, outfile, ts)
