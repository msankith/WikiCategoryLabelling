from DataParser import DataParser
from model2 import Model2 as Model


# In[ ]:

maxParagraphLength=250
maxParagraphs=10
labels=1000
vocabularySize=15000
model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training.getDataFromfile("data/vocab_3L_l1000_sampled_10000_red_train.txt")

batchSize=50

epoch=0
epochEnd=10
for e in range(epoch,epochEnd):
    print 'Epoch: ' + str(e)
    cost=0
    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
        #break
    print (str(cost))

    if e % 10 ==0:
        model.save("model2_l1000_"+str(e))
