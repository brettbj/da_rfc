"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.
 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).
 For the denoising autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]
 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""

import cPickle
import gzip
import os
import sys
import time
import csv

import numpy
import math

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from logistic_sgd import load_data
from utils import tile_raster_images

class dA(object):
    """Denoising Auto-Encoder class (dA)
    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.
    .. math::
        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
        y = s(W \tilde{x} + b)                                           (2)
        x = s(W' y  + b')                                                (3)
        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`
        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA
        :type n_visible: int
        :param n_visible: number of visible units
        :type n_hidden: int
        :param n_hidden:  number of hidden units
        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial
                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``
                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.
        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        # @TODO should we be using hard sigmoid or ultra fast sigmoid?
        hidden = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        return hidden

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        
        #switched cost function to |x-z
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L = T.sum(pow(abs(self.x - z), 2), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates, y)

def paramSweep():
    files = ['../data/bladderCancer_het.txt']
    hidden_layers = [1, 25, 50, 100, 200, 400, 800, 1200, 1422, 2844]
    lrs = [0.05]
    corruption_levels = [0.1, 0.2, 0.3]

    csvfile = open('results_dA.csv', 'wb')
    writer = csv.writer(csvfile)
    for f in files:
        print f
        #data = load_data.load_data(f)    
        for lr in lrs:
            for cl in corruption_levels:
                for hl in hidden_layers:
                    ret_vals = run_dA(dataset=f, hidden=hl, learning_rate=lr, corruption_level=cl)
                    print ret_vals
                    writer.writerow([ret_vals])
    csvfile.close()

def run_dA(training_epochs=100,
            dataset='../data/bladderCancer.txt',
            batch_size=10, output_folder='dA_plots',
            hidden=200, visi=1422, corruption_level=0.3, learning_rate=0.05):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder
    :type training_epochs: int
    :param training_epochs: number of epochs used for training
    
    :type dataset: string
    :param dataset: path to the picked dataset
    """
    datasets = load_data(dataset, random=True)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x') 

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #print "Build the model"
    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=visi, n_hidden=hidden)

    cost, updates, y = da.get_cost_updates(corruption_level=corruption_level,
                                        learning_rate=learning_rate)

    #print "Got Cost Updates"
    train_da = theano.function([index], cost, updates=updates,
         givens={x: train_set_x[index * batch_size:
                                  (index + 1) * batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    minCost = sys.maxint
    inc = 0
    i = 0
    epoch = 1
    while True:
        i += 1
        epoch = epoch + 1
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        if i>training_epochs:
        #if (numpy.mean(c) > minCost):
            inc = inc + 1
            #if (i>10):
            #if (inc >7 and i>250):
            break;
        else:
            inc = 0
            minCost = numpy.mean(c)

    # predict 
    a = T.dmatrix('a')
    get_hidden = da.get_hidden_values(a)
    f = theano.function([a], [get_hidden])
    train_set_x_hidden = f(train_set_x.get_value())
    test_set_x_hidden = f(test_set_x.get_value())

    rfc = RandomForestClassifier(n_estimators=100)
    case_percent_train = numpy.sum(train_set_y)/len(train_set_y)
    case_percent_test = numpy.sum(test_set_y)/len(test_set_y)

    w0 = 1-case_percent_train
    w1 = case_percent_train
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(train_set_x.get_value(), numpy.ravel(train_set_y), sample_weight=numpy.array([w0 if r==0 else w1 for r in train_set_y]))
    weighted_input = rfc.score(test_set_x.get_value(), numpy.ravel(test_set_y))

    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(train_set_x_hidden[0], numpy.ravel(train_set_y), sample_weight=numpy.array([w0 if r==0 else w1 for r in train_set_y]))
    weighted_hl = rfc.score(test_set_x_hidden[0], numpy.ravel(test_set_y))

    end_time = time.clock()
    da.minCost = minCost
    training_time = (end_time - start_time)
    print "Cost: ", da.minCost
    
    h = T.dmatrix('h')
    print hidden
    os.chdir('../')

    min_cost = da.minCost
    da = None
    # final hidden layer 
    return [weighted_input, weighted_hl]

if __name__ == '__main__':
    pre_rfc = []
    post_rfc = []
    for i in range(100):
        print i
        res = run_dA()
        pre_rfc.append(res[0])
        post_rfc.append(res[1])
        print pre_rfc
        print post_rfc