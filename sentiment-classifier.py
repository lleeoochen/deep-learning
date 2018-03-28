import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion = 0.1)

trainX, trainY = train
testX, testY = test

#Data preprocessing and sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#Conver labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

#Network building
net = tflearn.input_data([None, 100]) #Input Layer: Batch size and lengths
net = tflearn.embedding(net, input_dim=10000, output_dim=128) #Embedding Layer: Dimension of inputs and outpus
net = tflearn.lstm(net, 128, dropout=0.8) #LSTM Layer: Remember data from beginning of the sequence (improve prediction) and drop overfitting
net = tflearn.fully_connected(net, 2, activation='softmax') #Fully connected layer
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy') #Regression layer

#Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=2)
