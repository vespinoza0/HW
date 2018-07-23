import keras
import numpy as np
from numpy import linalg as LA
from numpy import genfromtxt
from keras.models import Sequential, Model
from keras import layers, Input, metrics
from keras.layers import Dense, Dropout, Embedding, Activation, LSTM
from keras.utils import to_categorical


def generator(x,y,batch):		# generator function, given a data set, yields batch of samples and corresponding targets
	iters = int(y.shape[0]/batch)
	print('number of batches',iters)
	y = to_categorical(y)		# one hot encoded targets
	while True:
		for i in range(iters):
			start= i*batch
			end =i*batch+batch
			samples = x[start:end,:]
			samples = l2Norm(samples)
			targets = y[start:end]
		yield samples,targets
		
def l2Norm(x):			# returns sequences normaized by L2 norm. In order to to prevent exploding gradient
	xx = np.zeros((x.shape[0],x.shape[1],1))
	for i in range(x.shape[0]):
		norm = LA.norm(x[i,:],ord=2)
		xx[i,:,0] =x[i,:]/norm
	return xx

if __name__ == "__main__":
	
	data = np.loadtxt('encodedTrainData.csv',delimiter=',')		#load train set
	y = data[:,0]				# train targets 
	x = data[:,1:]				# train sequences 
	del data					# delete data		
	batch = 20					# batch size
	eps = 20					# number of epochs to train
	SPE = x.shape[0]/batch 		# steps per epoch = batches in epochs
	c = x.shape[0]%batch		
	c = x.shape[0]- c			# cutoff number of sequences
	trainGen= generator(x[:c,:],y[:c],batch)		# generator function for train data
	
	data = np.loadtxt('encodedTestData.csv',delimiter=',')		#load test set
	y1 = data[:,0]						# test targets 
	x1 = data[:,1:]						# test sequences 
	del data 
	c = x1.shape[0]%batch		
	c = x1.shape[0]- c					# cutoff number of sequences
	y1 = to_categorical(y1[:c])			# one hot encode targets
	x1 = x1[:c,:]
	x1 = l2Norm(x1)                     # normalize test data
	
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, 	#generative LSTM layer
		batch_input_shape=(batch, 200, 1)))
	model.add(LSTM(16, return_sequences=False,dropout=0.3, recurrent_dropout=0.3))	#predictive LSTM layer
	model.add(Dense(16, activation='softmax'))										#softmax laer for 16 classes
	model.compile(loss='categorical_crossentropy', optimizer='adam', 
		metrics=['categorical_accuracy'])
		
	history = model.fit_generator(trainGen, steps_per_epoch=SPE,
			epochs=eps, validation_data=(x1,y1))
