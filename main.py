from DataParser import DataParser
from model import Model1_batch as Model


# In[ ]:

maxParagraphLength=250
maxParagraphs=10
labels=1000
vocabularySize=300000
model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
training.getDataFromfile("data/vocab_3L_l1000_red_train.txt")

#testing = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
#testing.getDataFromfile("data/vocab_3L_l1000_red_test.txt")
#model.load("./model1_W3L_l1000_6")
model.load("./model1_W3L_Alpha_5_batch_50_Exp2_l1000_75")
print ("-------------Training Begins-------------")
epoch=76
epochEnd=150
batchSize=50

f=open("trainingCost.log","w")
f.close()

for e in range(epoch,epochEnd):
    print 'Epoch: ' + str(e)
    cost=0
    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
        #break
    print (str(cost))
    print("Epoch ",e," Cost = ",cost)
    f=open("trainingCost.log","a")
    f.write("Epoch "+str(e)+" Cost = "+str(cost)+"\n")
    f.close()

    if e % 10 ==0:
        model.save("model1_W3L_Alpha_5_batch_50_Exp2_l1000_"+str(e))
