import numpy as np
import superpara
import gradient
import ann
import random
import adaptive
import chkweight
import trainset
import activation

def error(x, step):
	return np.square(gradient.temp_res3(x, step))

def restart():
	ann.weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE)
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array
	#chkweight.outweight()

def incweight(input, step): # step >= 1
	hidden_input = ann.PIPES[step - 1][0].dot(input)[:, 0] #the previous pipe weight ann
	hidden_out = activation.activation_fun(hidden_input)
	hidden_out_prime = activation.act_prime(hidden_input)
	weight_product = ann.PIPES[step - 1][0][:, -2] * ann.PIPES[step - 1][1] # weight_t_h * weight_h_o
	nn_out_prime = weight_product.dot(hidden_out_prime)
	delta_out = step * superpara.T_STEP * nn_out_prime
	delta_weight = - delta_out / hidden_out
		"""
		N_k(y_0, t_k) 
	return delta_weight

def gdescent(dataset, step):
	#errors of between two epochs
	error_pre = 0
	error_curr = 0
	error_delta = 0
	
	inc_weight_flag = True # define whether at the beginning of each time step, 
						   # adjust the weight when input the first training data
	
	for epoch in range(superpara.EPOCHS):
		#shuffle training data 
		random.shuffle(dataset)
		#dataset.reverse()

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

				# increment weight when meeting the first data of this step
				if inc_weight_flag:
					incweight(inputdata, step)
					inc_weight_flag = False	

				sum_grad_wyh += gradient.gradient_dw_y_h(inputdata, step)
				sum_grad_wth += gradient.gradient_dw_t_h(inputdata, step)
				sum_grad_wbh += gradient.gradient_dw_b_h(inputdata, step)
				sum_grad_who += gradient.gradient_dw_h_o(inputdata, step)

				#update error of this mini batch
				error_batch += error(inputdata, step)
				#print "\terror_batch", error_batch

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

			#chkweight.outweight()

			#update epoch error using error of this mini batch
			error_epoch += error_batch

		#update errors between two adjacent epochs
		error_pre = error_curr
		error_curr = np.sqrt(error_epoch / len(dataset) / 2.0)	#average error, mean and squared
		#error_curr = error_epoch 					#sum error
		error_delta = np.abs(error_curr - error_pre)

		#output
		print "error_curr:", error_curr, "error_pre:", error_pre, "error_delta:", error_delta, "rate: ", superpara.LEARN_RATE, "\n"


		if adaptive.jump(error_curr, error_delta) == 0:
			restart() #reset the weight matrix
			return 0  #restart from epoch 0

		adaptive.adjust(error_curr, error_delta) #adjust learning rate
		
		
	return 1 #all epochs finished, return 1, so the loop in itrdescent terminates

def itrdescent(dataset, step):
	termination = 0
	while termination != 1:
		try:
			termination = gdescent(dataset, step)
		except OverflowError:
			print "Overflow!!!"
			restart()
			termination = 0