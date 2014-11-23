import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from load_data import load_data

class LogisticRegression(object):

	# inputs
	# n_in number of input units - dimension of space in which the datapoints lie
	# number of output units
	def __init__(self, input, n_in, n_out):
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
		# compute vector of class-membership probabilities in symbolic form
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		# compute prediction as class whose probability is maximal in symbolic form
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		 # parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y): 
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred', ('y', target.type, 'y_pred', self.y_pred.type))

		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
        
		else:
			raise NotImplementedError()

# @TODO mnist.pkl.gz should be changed?
def sgd_optimization(learning_rate=0.05, n_epochs=1000, dataset='../data/csv_squences_AorBorALKorALL.txt', batch_size=20):
	datasets = load_data.load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	print '... building the model'
	print train_set_x

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	# need to get this from the dataset***
	classifier = LogisticRegression(input=x, n_in=1000, n_out=2)

	# need to minimize the negative log likelihood the model in symoblic format
	cost = classifier.negative_log_likelihood(y)
	
	# compiling a Theano function that computes the mistakes that are made by
	# the model on a minibatch
	test_model = theano.function(inputs=[index],
	        outputs=classifier.errors(y),
	        givens={
	            x: test_set_x[index * batch_size: (index + 1) * batch_size],
	            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

	validate_model = theano.function(inputs=[index],
	        outputs=classifier.errors(y),
	        givens={
	            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
	            y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

	# compute the gradient of cost with respect to theta = (W,b)
	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs.
	updates = [(classifier.W, classifier.W - learning_rate * g_W),
	       (classifier.b, classifier.b - learning_rate * g_b)]

	# compiling a Theano function `train_model` that returns the cost, but in
	# the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = theano.function(inputs=[index],
	    outputs=cost,
	    updates=updates,
	    givens={
	        x: train_set_x[index * batch_size:(index + 1) * batch_size],
	        y: train_set_y[index * batch_size:(index + 1) * batch_size]})

	print '... training the model'

	# early-stopping parameters
	patience = 5000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is found
	improvement_threshold = 0.995  # a relative improvement of this much is considered significant
	validation_frequency = min(n_train_batches, patience / 2)
	                          # go through this many
	                          # minibatche before checking the network
	                          # on the validation set; in this case we
	                          # check every epoch
	best_params = None
	best_validation_loss = numpy.inf
	test_score = 0.
	start_time = time.clock()

	done_looping = False
	epoch = 0

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i)
				                     for i in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
				    (epoch, minibatch_index + 1, n_train_batches,
				    this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
					    patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					#print 'update best validation ' + best_validation_loss
					# test it on the test set

					test_losses = [test_model(i)
					               for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of best'
					   ' model %f %%') %
					    (epoch, minibatch_index + 1, n_train_batches,
					     test_score * 100.))

			if patience <= iter:
			    done_looping = True
			    break

	end_time = time.clock()
	print(('Optimization complete with best validation score of %f %%,'
	       'with test performance %f %%') %
	             (best_validation_loss * 100., test_score * 100.))
	print 'The code run for %d epochs, with %f epochs/sec' % (
	    epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' +
	                      os.path.split(__file__)[1] +
	                      ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
	sgd_optimization()