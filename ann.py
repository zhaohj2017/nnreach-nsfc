import numpy as np
import superpara

weight_matrix = np.zeros((superpara.NUM_HIDDEN, superpara.INPUT_SIZE))

weight_y_h = weight_matrix[:, 0]    			#array (or matrix ?)
weight_t_h = weight_matrix[:, -2]    			#array
weight_b_h = weight_matrix[:, -1]    			#array
weight_h_o = np.zeros(superpara.NUM_HIDDEN)		#array


PIPES = [[weight_matrix, weight_h_o]]