import theano
import theano.tensor as T
import numpy
from numpy import genfromtxt

def load_data(datafile, random=True):
	data_file = datafile

	#print '... loading data'
	data = genfromtxt(data_file, delimiter='\t')

	if random:
		indices = range(1, 1283)
		numpy.random.shuffle(indices)

		#print data.shape
		train_set = data[indices[1:1100], :]
		#valid_set = data[indices[801:1000], :]
		test_set = data[indices[1101:1201], :]
		#print train_set
	else:
		train_set = data[1:1100, :]
		test_set = data[1101:1201, :]


	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		#data_x, data_y = data_xy
		data_x = data_xy[:, :-1]
		data_y = data_xy[:, -1]

		#print data_x.shape
		#print data_y.shape

		shared_x = theano.shared(numpy.asarray(data_x,
		                                       dtype=theano.config.floatX),
		                         borrow=borrow)
		# shared_y = theano.shared(numpy.asarray(data_y,
		#                                        dtype=theano.config.floatX),
		#                          borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, data_y # T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	#valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
	# rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	#         (test_set_x, test_set_y)]

	return rval