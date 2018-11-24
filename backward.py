import numpy as np
import superpara
import gradient
import ann
import random

def error(x):
	e = gradient.temp_res3(x)
	return e * e

def gdescent(dataset):

	error_pre = 0
	error_curr = 0
	error_delta = 0

	ann.weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE)
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array

	for epoch in range(superpara.EPOCHS):
		superpara.BATCH_NUM = len(dataset) / superpara.BATCH_SIZE
	
		if len(dataset) % superpara.BATCH_SIZE != 0:
			superpara.BATCH_NUM += 1
		#print "epoch:", epoch
		random.shuffle(dataset)
		error_epoch = 0
		for currbatch in range(superpara.BATCH_NUM):
			batchset = dataset[currbatch * superpara.BATCH_SIZE : (currbatch + 1) * superpara.BATCH_SIZE]
			sum_grad_wyh = np.zeros(len(ann.weight_y_h))
			sum_grad_wth = np.zeros(len(ann.weight_t_h))
			sum_grad_wbh = np.zeros(len(ann.weight_b_h))
			sum_grad_who = np.zeros(len(ann.weight_h_o))
			error_batch = 0
			for inputdata in batchset:
				sum_grad_wyh += gradient.gradient_dw_y_h(inputdata)
				sum_grad_wth += gradient.gradient_dw_t_h(inputdata)
				sum_grad_wbh += gradient.gradient_dw_b_h(inputdata)
				sum_grad_who += gradient.gradient_dw_h_o(inputdata)
				error_batch += error(inputdata)
			ann.weight_y_h += superpara.LEARN_RATE * sum_grad_wyh / len(batchset)
			ann.weight_t_h += superpara.LEARN_RATE * sum_grad_wth / len(batchset)
			ann.weight_b_h += superpara.LEARN_RATE * sum_grad_wbh / len(batchset)
			ann.weight_h_o += superpara.LEARN_RATE * sum_grad_who / len(batchset)
			error_epoch += error_batch
		error_pre = error_curr
		error_curr = error_epoch
		error_delta = np.abs(error_curr - error_pre)
		#print "error_curr:", error_curr, "error_pre:", error_pre, "error_delta:", error_delta, "rate: ", superpara.LEARN_RATE, "\n"


		if error_curr > 1e1 and error_delta < 1e-2:
			print "RESTARAT!!!\n"
			return 0
	
		if error_curr < 1e0 and error_delta < 1e0:
			superpara.LEARN_RATE = - 2
			superpara.BATCH_SIZE = 169
		
	return 1

def recdescent(dataset):
	termination = 0
	while termination != 1:
		termination = gdescent(dataset)
