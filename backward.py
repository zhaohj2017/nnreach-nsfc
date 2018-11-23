import numpy as np
import superpara
import gradient
import ann

def error(x):
	return np.abs(temp_res3(x))

def gdescent(x):
	epoch = 0
	error_pre = 0
	error_curr = 0
	delta_error = 0
	while epoch < superpara.EPOCHS:
		print "epoch:", epoch
		while x[0, 0] <= superpara.RANGE_Y[1]:
			while x[1, 0] <= superpara.RANGE_T[1]:
				ann.weight_y_h += superpara.LEARN_RATE * gradient.gradient_dw_y_h(x)
				ann.weight_t_h += superpara.LEARN_RATE * gradient.gradient_dw_t_h(x)
				ann.weight_b_h += superpara.LEARN_RATE * gradient.gradient_dw_b_h(x)
				ann.weight_h_o += superpara.LEARN_RATE * gradient.gradient_dw_h_o(x)
				x[1, 0] += superpara.MESH_SIZE_T
				"""
				error_pre = error_curr
				error_curr = error(x)
				delta_error = np.abs(error_curr - error_pre)
				print error_curr
				if delta_error < 1e-3:
					superpara.LEARN_RATE /= 2.0
				"""
			else:
				x[0, 0] += superpara.MESH_SIZE_Y
				x[1, 0] = superpara.RANGE_T[0]
		else:
			epoch += 1
			x[0, 0] = superpara.RANGE_Y[0]
