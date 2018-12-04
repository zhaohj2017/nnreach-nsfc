import numpy as np
import superpara

##set the initial weights
##standard normal distribution
#these are the working weights that are updated during the learning process
weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE)

weight_y_h = weight_matrix[:, 0]    			#array (or matrix ?)
weight_t_h = weight_matrix[:, -2]    			#array
weight_b_h = weight_matrix[:, -1]    			#array

weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array

#the neural network weight computed for the ith (i >= 0) step is stored at index i in PIPES
PIPES = []

