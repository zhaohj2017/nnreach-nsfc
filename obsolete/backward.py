import numpy as np
import superpara
import gradient
import ann
import random
import chkweight
import trainset
import activation

def error(w_matrix, w_ho, input, step): #square of the difference between derivatives: cost function, square error
	return np.square(gradient.temp_res3(w_matrix, w_ho, input, step))

def restart(): #reset the working weights
	ann.weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE) #matrix
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array
	#chkweight.outweight()

def gdescent(dataset, step):
	#errors of between two epochs
	error_pre = 0
	error_curr = 0
	error_delta = 0
	
	for epoch in range(superpara.EPOCHS):
		#shuffle training data 
		random.shuffle(dataset)
		#dataset.reverse() # (0, 0, Not good) (1, 0, Bad) (0, 1, Not good) (1, 1, Bad)

		#the number of mini batches
		superpara.BATCH_NUM = len(dataset) / superpara.BATCH_SIZE
		if len(dataset) % superpara.BATCH_SIZE != 0: #the remaining data, not a whole batch
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
			#is there a fast matrix operation for this loop?
			for inputdata in batchset:
				sum_grad_wyh += gradient.gradient_dw_y_h(ann.weight_matrix, ann.weight_h_o, inputdata, step)
				sum_grad_wth += gradient.gradient_dw_t_h(ann.weight_matrix, ann.weight_h_o, inputdata, step)
				sum_grad_wbh += gradient.gradient_dw_b_h(ann.weight_matrix, ann.weight_h_o, inputdata, step)
				sum_grad_who += gradient.gradient_dw_h_o(ann.weight_matrix, ann.weight_h_o, inputdata, step)

				#update error of this mini batch
				error_batch += error(ann.weight_matrix, ann.weight_h_o, inputdata, step)
				#print "\terror_batch", error_batch

			#update weight using gradient from this mini batch (average over the batch data)
			"""
			ann.weight_y_h += (- superpara.LEARN_RATE) * sum_grad_wyh			#sum gradient
			ann.weight_t_h += (- superpara.LEARN_RATE) * sum_grad_wth
			ann.weight_b_h += (- superpara.LEARN_RATE) * sum_grad_wbh
			ann.weight_h_o += (- superpara.LEARN_RATE) * sum_grad_who
			"""
			ann.weight_y_h += (- superpara.LEARN_RATE) * sum_grad_wyh / len(batchset)	#average gradient
			ann.weight_t_h += (- superpara.LEARN_RATE) * sum_grad_wth / len(batchset)
			ann.weight_b_h += (- superpara.LEARN_RATE) * sum_grad_wbh / len(batchset)
			ann.weight_h_o += (- superpara.LEARN_RATE) * sum_grad_who / len(batchset)

			#chkweight.outweight()

			#update epoch error using error of this mini batch
			error_epoch += error_batch

		#update errors between two adjacent epochs
		error_pre = error_curr
		error_curr = np.sqrt(error_epoch / len(dataset))	#average error, mean and squared
		error_delta = np.abs(error_curr - error_pre)

		#output
		print "error_curr:", error_curr, "error_pre:", error_pre, "error_delta:", error_delta, "rate: ", superpara.LEARN_RATE, "\n"

		if adaptive.jump(error_curr, error_delta) == 1:
			restart() #reset the weight matrix
			return 0  #restart from epoch 0

		adaptive.adjust(error_curr, error_delta) #adjust learning rate

		if adaptive.stop(error_curr, error_delta) == 1: #early stop
			return 1 #early return

	return 1 #all epochs finished, return 1, so the loop in itrdescent terminates

def itrdescent(dataset, step):
	termination = 0
	while termination != 1:
		try:
			termination = gdescent(dataset, step)
		except OverflowError: #will this work???
			print "Overflow!!!"
			restart()
			termination = 0




"""
##obsolete: increment weight is not helpful

if step > 0:
	incweight(input, step) # increment weight at the beginning of each time step
def incweight(input, step): # step >= 1
	hidden_input = ann.PIPES[step - 1][0].dot(input)[:, 0] #the previous pipe weight ann
	hidden_out = activation.activation_fun(hidden_input)
	hidden_out_prime = activation.act_prime(hidden_input)
	weight_product = ann.PIPES[step - 1][0][:, -2] * ann.PIPES[step - 1][1] # weight_t_h * weight_h_o
	nn_out_prime = weight_product.dot(hidden_out_prime)
	delta_out = superpara.T_STEP * nn_out_prime
	delta_weight = delta_out / sum(hidden_out) * (np.zeros(len(ann.weight_h_o)) + 1)
	ann.weight_h_o += delta_weight
"""