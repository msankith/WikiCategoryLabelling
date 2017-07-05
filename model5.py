
import tensorflow as tf
import math

import numpy as np

class Model5:
    def __init__(self,maxParagraphLength,maxParagraphs,labels,vocabularySize,labelsActive,batchSize):
        '''
        Constructor
        '''
        self.wordEmbeddingDimension = 50
        self.labelEmbeddingDimension=500
        self.labelsActive=labelsActive

        self.vocabularySize=vocabularySize
        self.labels=labels
        self.filterSizes_paragraph = [2,3,4]
        self.paragraphLength=maxParagraphLength
        self.num_filters_parargaph=40
        self.maxParagraph = maxParagraphs
        self.poolLength=3
        
        self.paragraphOutputSize = len(self.filterSizes_paragraph)*self.num_filters_parargaph*int(math.ceil(maxParagraphLength/float(self.poolLength)))
        self.device ='gpu'
        self.wordEmbedding = tf.get_variable("word_Embedding",shape=[self.vocabularySize, self.wordEmbeddingDimension],dtype=tf.float32)
        self.labelEmbedding = tf.get_variable("label_Embedding",shape=[self.labels+5,self.labelEmbeddingDimension],dtype=tf.float32)
        
        self.batch_size=batchSize
        self.lstm_state_size=400
        self.labelProjectionMatrix=tf.Variable(tf.random_uniform([ self.lstm_state_size,self.labelEmbeddingDimension], -1.0, 1.0),name="labelProjection")
        self.paragraphProjectionMatrix=tf.Variable(tf.random_uniform([self.paragraphOutputSize,self.labelEmbeddingDimension], -1.0, 1.0),name="labelProjection")
        self.attentionScoreMatrix = tf.Variable(tf.truncated_normal([self.lstm_state_size,self.paragraphOutputSize], stddev=0.1),name="AttentionScoreMatrix")
        
        self.paragraphList = []
        for i in range(self.maxParagraph):
            self.paragraphList.append(tf.placeholder(tf.int32,[None,self.paragraphLength],name="paragraphPlaceholder"+str(i)))
        
        self.labelsPlaceholder =[]
        self.labelsMask=[]
        for i in range(self.labelsActive):
            self.labelsPlaceholder.append(tf.placeholder(tf.int32,[self.batch_size]))
            self.labelsMask.append(tf.placeholder(tf.float32,[self.batch_size,self.labels+5]))
        
        
        
        self.graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
    
    def graph(self):
        device_name=self.device
        with tf.device(device_name): 
            self.prediction=[]
            self.paraEmbedding=self.allParagraphEmbedding(self.paragraphList,self.filterSizes_paragraph,self.num_filters_parargaph)
            lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_state_size)
            
            self.lables_probability_all=[]
            labelAtT= [self.labels+1]*self.batch_size
            self.lstm_state=(tf.zeros([self.batch_size, lstm.state_size.c]),)*2
            self.lstm=lstm
                
            
            for i in range(self.labelsActive):
                self.currentLabEmbedding = tf.nn.embedding_lookup(self.labelEmbedding,labelAtT)
                if i > 0:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        tf.get_variable_scope().reuse_variables()
                        self.lstm_output, self.lstm_state = lstm(self.currentLabEmbedding, self.lstm_state)
                else:
                    self.lstm_output, self.lstm_state = lstm(self.currentLabEmbedding, self.lstm_state)
                self.alpha=self.alphaVector(self.lstm_output,self.paraEmbedding)
                self.context= self.contextVector(self.alpha,self.paraEmbedding)
                self.VectorAtT=self.projectionOnLabelSpace(self.lstm_output,self.context)
                self.labelProbability,self.labelAtT=self.getLabelAtT(self.VectorAtT)
                self.lables_probability_all.append(self.labelProbability)
                self.prediction.append(self.labelAtT)
            
            self.lables_probability_all=tf.multiply(self.lables_probability_all,self.labelsMask)
            with tf.device(None):
                self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.lables_probability_all,labels=self.labelsPlaceholder))
 
        #     self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        # self.lables_probability_all,        self.labelsPlaceholder,
        # [tf.ones([self.batch_size], dtype=tf.float32)]*self.labelsActive)
        

            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(self.cost)
        
    
    def alphaVector(self,rnnOutput, paragraphEmbedding):
        
        self.tempAttention= tf.matmul(rnnOutput,self.attentionScoreMatrix)
        
        alphas=[]
        for para in paragraphEmbedding:
            self.adaProduct=tf.multiply(self.tempAttention, para)
            paraAlpha=tf.reduce_sum(self.adaProduct,axis=1)
            alphas.append(tf.expand_dims(paraAlpha,1))
        alphas = tf.nn.softmax(tf.concat(alphas,axis=1))
        return alphas
    
    def contextVector(self,alphas,paragraphEmbedding):
        self.alphasRe=tf.reshape(alphas,[self.batch_size,1,self.maxParagraph])
        temp=[]
        for para in paragraphEmbedding:
            temp.append(tf.reshape(para,[-1,1,self.paragraphOutputSize]))
        allPara = tf.concat(temp,axis=1)
        contextV=tf.matmul(self.alphasRe,allPara)
        return contextV
    
    def projectionOnLabelSpace(self,lstm_output,contextVector):
        self.labProjection=tf.matmul(lstm_output,self.labelProjectionMatrix)
        contextVector=tf.reshape(contextVector,[self.batch_size,self.paragraphOutputSize])
        self.paraProjection=tf.matmul(contextVector,self.paragraphProjectionMatrix)
        
        
        return tf.add(self.labProjection,self.paraProjection)
    
    def getLabelAtT(self,contextVector):
        # labelProbaility = tf.nn.softmax(tf.matmul(contextVector,tf.transpose(self.labelEmbedding)))
        labelProbaility = tf.matmul(contextVector,tf.transpose(self.labelEmbedding))
        labelPrediction= tf.argmax(labelProbaility,axis=1)
        return labelProbaility,labelPrediction
    
    def getParagraphEmbedding(self,paragraphWords):
        device_name="/cpu:0"
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            paraEmbedding=tf.nn.embedding_lookup(self.wordEmbedding,paragraphWords)
        return tf.expand_dims(paraEmbedding, -1)
    
    def allParagraphEmbedding(self,paragraphVectorList,filterSizes_paragraph,num_filters_parargaph):
        paragraphCNNEmbedding=[]
        for paragraph in paragraphVectorList:
            paragraphVector = self.getParagraphEmbedding(paragraph)
            cnnEmbedding = self.convLayeronParagraph(paragraphVector,filterSizes_paragraph,1,num_filters_parargaph)
            paragraphCNNEmbedding.append(tf.reshape(cnnEmbedding,[-1,self.paragraphOutputSize]))
        return paragraphCNNEmbedding
   
    
    
    def convLayeronParagraph(self,paragraphVector,filterSizes,num_input_channels,num_filters):
        pooled_outputs=[]
        for filter_size in filterSizes:
            shape = [filter_size,self.wordEmbeddingDimension,1,num_filters]

            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="paragraphConvLayerW_"+str(filter_size))
            bias= tf.Variable(tf.constant(0.1, shape=[num_filters]),name="paragraphConvLayerB_"+str(filter_size))
            conv = tf.nn.conv2d(
                        paragraphVector,
                        weights,
                        strides=[1, 1, self.wordEmbeddingDimension, 1],
                        padding="SAME",
                        name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            pool_length=self.poolLength
            pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, pool_length, 1, 1],
                        strides=[1, pool_length, 1, 1],
                        padding='SAME',
                        name="pool")
            pooled_outputs.append(pooled)
        return tf.concat(pooled_outputs,axis=1)

     
    def train(self,data):
        feed_dict_input={}
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
        for p in range(self.labelsActive):
            feed_dict_input[self.labelsPlaceholder[p]]=data[0][p]
            feed_dict_input[self.labelsMask[p]]=data[2][p]
        _, cost = self.session.run((self.optimizer,self.cost),feed_dict=feed_dict_input)
        return cost

    def predict(self,data):
        feed_dict_input={}
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
        for p in range(self.labelsActive):
            feed_dict_input[self.labelsPlaceholder[p]]=data[0][p]
        pred=self.session.run(self.prediction,feed_dict=feed_dict_input)
        return pred
    
    def getError(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
        cost = self.session.run(self.cost,feed_dict=feed_dict_input)
        return cost 

    def save(self,save_path):
        saver = tf.train.Saver()
        saver.save(self.session, save_path)


    def load(self,save_path):
        self.session = tf.Session()
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, save_path)


    def save_label_embeddings(self):
        pass

    

