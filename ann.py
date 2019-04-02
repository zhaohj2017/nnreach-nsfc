import numpy as np
import superpara

##set the initial weights
##standard normal distribution
#these are the working weights that are updated during the learning process

#why does not work well??? very bad
#good or bad, try to test, more!!!
#rand_sigma = 1.0 / np.sqrt((superpara.INPUT_SIZE + 1) * superpara.NUM_HIDDEN)
rand_mu = 0
rand_sigma = 1
weight_matrix = rand_sigma * np.random.randn(superpara.NUM_HIDDEN, superpara.INPUT_SIZE) + rand_mu

weight_y_h = weight_matrix[:, 0]    			#array (or matrix ?)
weight_t_h = weight_matrix[:, -2]    			#array
weight_b_h = weight_matrix[:, -1]    			#array

weight_h_o = rand_sigma * np.random.randn(superpara.NUM_HIDDEN) + rand_mu	#array

#velocity and momentum
velocity = np.zeros([len(weight_y_h) + len(weight_t_h) + len(weight_b_h) + len(weight_h_o), 1])


def outweight():
	global weight_matrix
	global weight_y_h
	global weight_t_h
	global weight_b_h
	global weight_h_o

	print "The weight matrix"
	print weight_matrix.T

	print "The weight w.r.t. each input state"
	print "weight-input-y-hidden"
	print weight_y_h
	print "weight-input-t-hidden"
	print weight_t_h
	print "bias"
	print weight_b_h

	print "weight-hidden-output"
	print weight_h_o

	print ""
	