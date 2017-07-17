from DataParser_wiki10 import DataParser_wiki10 as DataParser
from model2_wiki import Model2_wiki as Model


# In[ ]:

maxParagraphLength=2500
maxParagraphs=1
#nlabels=1001
#vocabularySize=76391
vocabularySize=101939
nlabels=30938
model = Model(maxParagraphLength,maxParagraphs,nlabels,vocabularySize)
training = DataParser(maxParagraphLength,nlabels,vocabularySize)
training.getDataFromfile("Wiki10/wiki10_train.pkl")
#training.getDataFromfile("data/wiki_fea_76390_Label_1000_train")

batchSize=50

epoch=0
epochEnd=100
for e in range(epoch,epochEnd):
    print 'Epoch: ' + str(e+1)
    cost=0
    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
    print (str(cost/training.totalPages))

    if (e+1)%10 == 0:
        print 'saving model..'
        model.save("model2_wiki10_"+str(e+1))
