import time
import random
import numpy as np
np.seterr(all = 'ignore')

# transfer functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
  return y * (1.0 - y)

# using tanh over logistic sigmoid for the hidden layer is recommended  
def tanh(x):
  return np.tanh(x)

# derivative for tanh sigmoid
def dtanh(y):
  return 1 - y*y

class MultiLayerPerceptron(object):
  """
  Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
  Consists of three layers: input, hidden and output. The sizes of input and output must match data
  the size of hidden is user defined when initializing the network.
  The algorithm can be used on any dataset.
  As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                      [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                      ...
                      [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
  An example is provided below with the digit recognition dataset provided by sklearn
  Fully pypy compatible.
  """
  def __init__(self, input, hidden, output, iterations=50, learning_rate=0.01, 
        rate_decay=1.0, verbose=True, wi_file=None, wo_file=None):
    """
    :param input: number of input neurons
    :param hidden: number of hidden neurons
    :param output: number of output neurons
    :param iterations: how many epochs
    :param learning_rate: initial learning rate
    :param rate_decay: how much to multiply learning rate by on each iteration (epoch)
    :param verbose: whether to spit out error rates while training
    :param wi_file: name of npy file to read input weights from
    :param wo_file: name of npy file to read output weights from
    """
    # initialize parameters
    self.iterations = iterations
    self.learning_rate = learning_rate
    self.rate_decay = rate_decay
    self.verbose = verbose

    # initialize arrays
    self.input = input + 1 # add 1 for bias node
    self.hidden = hidden 
    self.output = output

    # set up array of 1s for activations
    self.ai = np.ones(self.input)
    self.ah = np.ones(self.hidden)
    self.ao = np.ones(self.output)

    if wi_file is not None and wo_file is not None:
     self.wi = np.load(wi_file)
     self.wo = np.load(wo_file)
    else:
     self.randomize()

  def randomize(self, start=0, step=1):
    ''' create randomized weights
    use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
    variable entropy can be introduced to existing arrays using the start and step arguments
    '''
    input_range = 1.0 / self.input ** (1/2)
    wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
    wo = np.random.uniform(size=(self.hidden, self.output)) / np.sqrt(self.hidden)
    if step == 1:
      self.wi = wi
      self.wo = wo
    else: # NB arrays must already exist for this to work
      self.wi[start::step] = wi[start::step]
      self.wo[start::step] = wo[start::step]

  def feed_forward(self, inputs):
    """
    The feed_forward algorithm loops over all the nodes in the hidden layer and
    adds together all the outputs from the input layer * their weights
    the output of each node is the sigmoid function of the sum of all inputs
    which is then passed on to the next layer.
    :param inputs: input data
    :return: updated activation output vector
    """
    if len(inputs) != self.input-1:
      raise ValueError('Wrong number of inputs you silly goose!')

    ''' input activations '''
    self.ai[0:self.input -1] = inputs
    ''' hidden activations (einsum faster than dot if BLAS not installed) '''
    #self.ah = tanh(np.einsum('i,ij', self.ai, self.wi))
    self.ah = tanh(np.dot(self.ai, self.wi))
    ''' output activations (einsum faster than dot if BLAS not installed) '''
    #self.ao = sigmoid(np.einsum('i,ij', self.ah, self.wo))
    self.ao = sigmoid(np.dot(self.ah, self.wo))

    return self.ao

  def back_propagate(self, targets, learning_rate=None):
    """
    For the output layer
    1. Calculates the difference between output value and target value
    2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
    3. update the weights for every node based on the learning rate and sigmoid derivative

    For the hidden layer
    1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
    2. get derivative to determine how much weights need to change
    3. change the weights based on learning rate and derivative
    :param targets: y values
    :param N: learning rate
    :return: updated weights
    """
    if len(targets) != self.output:
      raise ValueError('Wrong number of targets you silly goose!')
    if learning_rate is not None:
      self.learning_rate = learning_rate

    ''' calculate error terms for output
     the delta (theta) tells you which direction to change the weights '''
    output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
    ''' calculate error terms for hidden (einsum faster than dot if BLAS not installed)'''
    #hidden_deltas = dtanh(self.ah) * np.einsum('ij,j', self.wo, output_deltas)
    hidden_deltas = dtanh(self.ah) * np.dot(self.wo, output_deltas)
    ''' update the weights connecting hidden to output '''
    self.wo -= self.learning_rate * output_deltas * self.ah.reshape(self.hidden, 1)
    ''' update the weights connecting input to hidden '''
    self.wi -= self.learning_rate * hidden_deltas * self.ai.reshape(self.input, 1)
    ''' calculate error '''
    error = sum(0.5 * (targets - self.ao)**2)
    return error

  def test(self, patterns):
    """
    Currently this will print out the targets next to the predictions.
    Not useful for actual ML, just for visual inspection.
    """
    nright = 0
    for p in patterns:
      x = np.where(p[1] > 0.5)
      y = np.where(self.feed_forward(p[0]) > 0.5)
      try:
        if len(x) > 0 and len(y) > 0 and x[0] == y[0]:
         nright += 1
      except:
        print('odd array from np.where', x, y)
    print('{:3.1f}% right'.format(100.0 * nright / len(patterns)))

  def train(self, patterns):
    if self.verbose == True:
      print('Using logistic sigmoid activation in output layer')

    num_example = np.shape(patterns)[0]
    for j in range(10):
      last_error = None
      error_trend = 500.0
      self.randomize(j, 7)
      last_tm = time.time() - 60.0
      print("-------------------- randomized -----------------")
      for i in range(self.iterations):
        error = 0.0
        random.shuffle(patterns)
        for p in patterns:
          inputs = p[0]
          targets = p[1]
          self.feed_forward(inputs)
          error += self.back_propagate(targets)

        if last_error is None:
          last_error = error
        error_trend = error_trend * 0.75 + (last_error - error) * 0.25
        last_error = error

        tm = time.time()
        if self.verbose == True and tm > (last_tm + 10.0):
          print('{} {:5.1f} Training error {:6.6f}, trend {:6.4f}'.format(i, tm - last_tm, error / num_example, error_trend / error))
          if (error_trend / error) < 0.05 and i > 20:
            break
          last_tm = tm

        # learning rate decay
        self.learning_rate *= self.rate_decay
        if error < (0.00025 * num_example):
          return

  def predict(self, X):
    """
    return list of predictions after training algorithm
    """
    predictions = []
    for p in X:
      predictions.append(self.feed_forward(p))
    return predictions
