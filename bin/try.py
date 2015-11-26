import numpy as np
from keras.models import Sequential
from seq2seq.seq2seq import Seq2seq


vocab_size = 10 #number of words
maxlen = 3 #length of input sequence and output sequence
embedding_dim = 5 #word embedding size
hidden_dim = 50 #memory size of seq2seq
batch_size = 7

seq2seq = Seq2seq(input_length=maxlen,
                  input_dim=embedding_dim,
                  hidden_dim=hidden_dim,
                  output_dim=vocab_size,
                  output_length=maxlen,
                  batch_size=batch_size,
                  depth=1)

print 'Build model ...'
model = Sequential()
model.add(seq2seq)
model.compile(loss='mse', optimizer='adam')

print 'Generate dummy data ...'
train_examples_num = batch_size
X = np.zeros((train_examples_num, maxlen, embedding_dim))
Y = np.zeros((train_examples_num, maxlen, vocab_size))

for train_example_idx in xrange(train_examples_num):
    for word_idx in xrange(maxlen):
        w2v_vector = np.random.rand(1, embedding_dim)[0]
        X[train_example_idx][word_idx] = w2v_vector

        bool_vector = np.zeros(vocab_size)
        bool_vector[np.random.choice(vocab_size)] = 1
        Y[train_example_idx][word_idx] = bool_vector

print X.shape, X
print Y.shape, Y

print 'Fit data ...'
model.fit(X, Y)
