import numpy as np
import superpara
import gradient
import ann

def error(x):
	return np.abs(temp_res3(x))

def gdescent(dataset):
	error_pre = 0
	error_curr = 0
	delta_error = 0
	for epoch in range(superpara.EPOCHS):
		print "epoch:", epoch
		for inputdata in dataset:
			ann.weight_y_h += superpara.LEARN_RATE * gradient.gradient_dw_y_h(inputdata)
			ann.weight_t_h += superpara.LEARN_RATE * gradient.gradient_dw_t_h(inputdata)
			ann.weight_b_h += superpara.LEARN_RATE * gradient.gradient_dw_b_h(inputdata)
			ann.weight_h_o += superpara.LEARN_RATE * gradient.gradient_dw_h_o(inputdata)
			"""
			error_pre = error_curr
			error_curr = error(inputdata)
			delta_error = np.abs(error_curr - error_pre)
			print error_curr
			if delta_error < 1e-3:
				superpara.LEARN_RATE /= 2.0
			"""
