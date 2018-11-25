import numpy as np
import superpara
import gradient
import ann
import random

def error(x):
	e = gradient.temp_res3(x)
	return e * e

def restart():
	ann.weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE)
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array
	
def gdescent(dataset):
	#errors of between two epochs
	error_pre = 0
	error_curr = 0
	error_delta = 0

	#reset network weights
	restart()

	for epoch in range(superpara.EPOCHS):
		#shuffle training data 
		random.shuffle(dataset)

		#the number of mini batches
		superpara.BATCH_NUM = len(dataset) / superpara.BATCH_SIZE
		if len(dataset) % superpara.BATCH_SIZE != 0:
			superpara.BATCH_NUM += 1

		#error of current epoch
		error_epoch = 0
		print "epoch:", epoch
		
		for currbatch in range(superpara.BATCH_NUM):
			#data set of this mini batch
			#print "\tmini_batch:", currbatch
			batchset = dataset[currbatch * superpara.BATCH_SIZE : (currbatch + 1) * superpara.BATCH_SIZE]

			#sum of gradient of this mini batch
			sum_grad_wyh = np.zeros(len(ann.weight_y_h))
			sum_grad_wth = np.zeros(len(ann.weight_t_h))
			sum_grad_wbh = np.zeros(len(ann.weight_b_h))
			sum_grad_who = np.zeros(len(ann.weight_h_o))

			#error of this mini batch
			error_batch = 0

			#update gradient using data from this mini batch
			for inputdata in batchset:
				sum_grad_wyh += gradient.gradient_dw_y_h(inputdata)
				sum_grad_wth += gradient.gradient_dw_t_h(inputdata)
				sum_grad_wbh += gradient.gradient_dw_b_h(inputdata)
				sum_grad_who += gradient.gradient_dw_h_o(inputdata)

				#update error of this mini batch
				error_batch += error(inputdata)

			#update weight using gradient from this mini batch (average over the batch data)
			"""
			ann.weight_y_h += superpara.LEARN_RATE * sum_grad_wyh			#sum gradient
			ann.weight_t_h += superpara.LEARN_RATE * sum_grad_wth
			ann.weight_b_h += superpara.LEARN_RATE * sum_grad_wbh
			ann.weight_h_o += superpara.LEARN_RATE * sum_grad_who
			"""
			ann.weight_y_h += superpara.LEARN_RATE * sum_grad_wyh / len(batchset)	#average gradient
			ann.weight_t_h += superpara.LEARN_RATE * sum_grad_wth / len(batchset)
			ann.weight_b_h += superpara.LEARN_RATE * sum_grad_wbh / len(batchset)
			ann.weight_h_o += superpara.LEARN_RATE * sum_grad_who / len(batchset)


			#update epoch error using error of this mini batch
			error_epoch += error_batch

		#update errors between two adjacent epochs
		error_pre = error_curr
		error_curr = np.sqrt(error_epoch / len(dataset) / 2.0)	#average error, mean and squared
		#error_curr = error_epoch 					#sum error
		error_delta = np.abs(error_curr - error_pre)

		#output
		print "error_curr:", error_curr, "error_pre:", error_pre, "error_delta:", error_delta, "rate: ", superpara.LEARN_RATE, "\n"


		#adjust learn rate or restart
		if error_curr > 1e0 and error_delta < 1e-2:
			print "RESTARAT!!!\n"
			return 0 #return from gradescent early, 0 means that the loop in itrdescent continues, but stop a new loop (restart)
	
		
		if error_curr < 1e-1 and error_delta < 1e-2: #continue to the next epoch
			superpara.LEARN_RATE = - 1
			superpara.BATCH_SIZE = 1
		
		
	return 1 #all epochs finished, return 1, so the loop in itrdescent terminates

def itrdescent(dataset):
	termination = 0
	while termination != 1:
		termination = gdescent(dataset)
