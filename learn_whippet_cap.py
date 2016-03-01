import time
import random
import numpy as np
np.seterr(all = 'ignore')

from perceptron import MultiLayerPerceptron


def load_data():
    data = np.loadtxt('Data/learndb.csv', delimiter = ',')

    # first five values are the one hot encoded y (target) values
    y = data[:, :5] # output results picture category
    x = data[:, 5:] # inputs histogram of gradients
    return [[x[i], y[i]] for i in range(data.shape[0])]

LEARN = True # control saving to weight files. Must do before setting False
start = time.time()
X = load_data()

wif = None if LEARN else 'wi_file.npy'
wof = None if LEARN else 'wo_file.npy'

NN = MultiLayerPerceptron(2048, 64, 5, iterations = 50, learning_rate = 0.01, 
                    rate_decay = 0.9999, wi_file=wif, wo_file=wof)
if LEARN:
  NN.train(X)
  np.save(wif, NN.wi)
  np.save(wof, NN.wo)
  end = time.time()
  print('Weights saved. Learning took {:4.1f}s'.format(end - start))
  print('To re-use these weights change LEARN above to False')

NN.test(X)
