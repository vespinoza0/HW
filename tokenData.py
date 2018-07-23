import numpy as np 
import csv
import keras
from keras import preprocessing
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
import pickle
import matplotlib.pyplot as plt
import itertools


def SeqEncoder(tokenizer, data): 		## create word mapping 
	words =[]
	print('creating list of words..')
	for i in range(len(data)):
		seq = data[i][1]
		seq = seq.split()
		words.append(seq)
	print('out of for loop')
	words = list(itertools.chain.from_iterable(words))
	print('tokenizing words...')
	tokenizer.fit_on_texts(words)
	print('ok,done.')
	del words
	return tokenizer
	
def targetEncoder(tokenizer, data): 		## create word mapping 
	targets =[]
	print('creating list of target words..')
	for i in range(len(data)):
		target = data[i][0]
		targets.append(target)
	print('tokenizing targets...')
	tokenizer.fit_on_texts(targets)
	print('ok,done.')
	del targets 
	return tokenizer
	
def encodeData(tokenizer,seqs):		# map sentence to a sentence of integers 
	xx = [] 				# list 
	print('encoding data...')
	for i in range(len(seqs)):
		seq = seqs[i][1]
		words = seq.split()
		sentence = tokenizer.texts_to_sequences(words)		     # map words to integer in dictionary
		sentence = list(itertools.chain.from_iterable(sentence)) # flatten sentence 
		xx.append(sentence)									     # put in list
	return xx

def encodeTargets(tokenizer, targets):  # map Targets to integers, return list of integers that represent class label
	yy = []				
	print('encoding targets ...')
	for i in range(len(targets)):
		out = targets[i][0]
		out1 = tokenizer.texts_to_sequences([out])
		yy.append(int(out1[0][0]))
	return yy

def newCSV(x,y,name):
	encoded = np.zeros((len(x),len(x[0])+1))
	for i in range(len(x)):
		encoded[i,0] = int(y[i])
		encoded[i,1:] = x[i]
		
	np.savetxt(name,encoded,delimiter=",")



if __name__ == "__main__":
	
	with open('shuffled-full-set-hashed.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile,delimiter=',')
		data = list(csv.reader(csvDataFile))
		
	tokenizer = Tokenizer(split=',') 			# tokenizer for sequnces
	targetTokenizer  = Tokenizer(split=',') 	# tokenizer for targets
	
	numOfseqs = int(len(data)*0.7)		#split training data 70%
	print('Num of seqs', numOfseqs)
	maxLen = 200
	tokenizer = SeqEncoder(tokenizer,data[0:numOfseqs])					# create dictionary for sequences
	targetTokenizer = targetEncoder(targetTokenizer, data[0:numOfseqs]) # create dictionary for targets
	
	xx = encodeData(tokenizer,data[0:numOfseqs])			# encode sequences to integers
	yy = encodeTargets(targetTokenizer,data[0:numOfseqs])	# encode targets to integers
	
	with open('SeqTokenizer.pickle', 'wb') as handle:				# save sequence and target tokenizers
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('TargetTokenizer.pickle', 'wb') as handle:
		pickle.dump(targetTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print('zero padding')
	x_train = preprocessing.sequence.pad_sequences(xx, maxlen=maxLen, truncating='post',padding='post')
	print('x_train',len(x_train),len(x_train[0]))
	newCSV(x_train, yy, 'encodedTrainData.csv')
	
	## test data 
	xx = encodeData(tokenizer,data[numOfseqs:])				# encode sequences to integers TEST SET 
	yy = encodeTargets(targetTokenizer,data[numOfseqs:])	# encode targets to integers TEST SET
	
	print('zero padding')
	x_train = preprocessing.sequence.pad_sequences(xx, maxlen=maxLen, truncating='post',padding='post')
	print('x_test',len(x_train),len(x_train[0]))
	newCSV(x_train, yy, 'encodedTestData.csv')
		
	