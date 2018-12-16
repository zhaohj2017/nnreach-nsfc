import superpara
import gradient
import numpy as np
import ann
import random
import adaptive
import chkweight
import trainset
import activation

def error(x, step): #square of the difference between derivatives: cost function, square error
	return np.square(gradient.temp_res3(x, step))

def restart(): #reset the working weights
	ann.weight_matrix = np.random.rand(superpara.NUM_HIDDEN, superpara.INPUT_SIZE) #matrix
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.rand(superpara.NUM_HIDDEN)	#array
	#chkweight.outweight()


#global loop variables for bfgs quasi-newton 
HESSE_SIZE = (superpara.INPUT_SIZE + 1) * superpara.NUM_HIDDEN
m_hesse = np.eye(HESSE_SIZE)
delta_parameter = np.zeros([HESSE_SIZE, 1])
pre_gradient = np.zeros([HESSE_SIZE, 1])

def gdescent(dataset, step):
	#errors of between two epochs
	error_pre = 0
	error_curr = 0
	error_delta = 0
	
	#loop variables for bfgs quasi-newton 
	global HESSE_SIZE
	global m_hesse
	global delta_parameter
	global pre_gradient

	for epoch in range(superpara.EPOCHS):
		#shuffle training data 
		random.shuffle(dataset)

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

			#sum of gradient for this mini batch
			sum_grad_wyh = np.zeros(len(ann.weight_y_h))
			sum_grad_wth = np.zeros(len(ann.weight_t_h))
			sum_grad_wbh = np.zeros(len(ann.weight_b_h))
			sum_grad_who = np.zeros(len(ann.weight_h_o))

			#error for this mini batch
			error_batch = 0

			#update weights using data from this mini batch
			#a bfgs quasi-newton implementation, rather than gradient descent
			for inputdata in batchset:
				sum_grad_wyh += gradient.gradient_dw_y_h(inputdata, step)
				sum_grad_wth += gradient.gradient_dw_t_h(inputdata, step)
				sum_grad_wbh += gradient.gradient_dw_b_h(inputdata, step)
				sum_grad_who += gradient.gradient_dw_h_o(inputdata, step)

				#update error of this mini batch
				error_batch += error(inputdata, step)
				#print "\terror_batch", error_batch

            #compute the average gradient for this mini batch
			avg_gradient = np.append(sum_grad_wyh / len(batchset), sum_grad_wth / len(batchset))
			avg_gradient = np.append(avg_gradient, sum_grad_wbh / len(batchset))
			avg_gradient = np.append(avg_gradient, sum_grad_who / len(batchset))
			avg_gradient = avg_gradient.reshape(avg_gradient.size, 1)

			#update delta gradient
			delta_gradient = avg_gradient - pre_gradient
			#update pre gradient
			pre_gradient = avg_gradient

			#update rho
			if delta_gradient[:, 0].dot(delta_parameter[:, 0]) == 0:
				rho = 0
			else:
				rho = 1.0 / delta_gradient[:, 0].dot(delta_parameter[:, 0])

			#first update hesse matrix (for the initial loop, delta_parameter = 0, so m_hesse is identity)
			hesse_temp1 = np.eye(HESSE_SIZE) - rho * np.dot(delta_parameter, delta_gradient.T)
			hesse_temp2 = np.dot(hesse_temp1, m_hesse)
			hesse_temp3 = np.eye(HESSE_SIZE) - rho * np.dot(delta_gradient, delta_parameter.T)
			hesse_temp4 = np.dot(hesse_temp2, hesse_temp3)
			m_hesse = hesse_temp4 + rho * np.dot(delta_parameter, delta_parameter.T)

			#then update direction
			direction = m_hesse.dot(avg_gradient)

			#then update learn rate
			pass

			#update delta parameter
			delta_parameter = (- superpara.LEARN_RATE) * direction

			#then update weights
			ann.weight_y_h += delta_parameter[0 : superpara.NUM_HIDDEN, 0] #extract
			ann.weight_t_h += delta_parameter[superpara.NUM_HIDDEN : superpara.NUM_HIDDEN * 2, 0]
			ann.weight_b_h += delta_parameter[superpara.NUM_HIDDEN * 2 : superpara.NUM_HIDDEN * 3, 0]
			ann.weight_h_o += delta_parameter[superpara.NUM_HIDDEN * 3 : direction.size, 0]


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
			m_hesse = np.eye(HESSE_SIZE)
			delta_parameter = np.zeros([HESSE_SIZE, 1])
			pre_gradient = np.zeros([HESSE_SIZE, 1])
			restart() #reset the weight matrix
			return 0  #restart from epoch 0

		adaptive.adjust(error_curr, error_delta) #adjust learning rate

		if adaptive.stop(error_curr, error_delta) == 1: #early stop
			return 1 #early return

	return 1 #all epochs finished, return 1, so the loop in itrdescent terminates

def itrdescent(dataset, step):
	#loop variables for bfgs quasi-newton 
	global HESSE_SIZE
	global m_hesse
	global delta_parameter
	global pre_gradient
	termination = 0

	while termination != 1:
		try:
			termination = gdescent(dataset, step)
		except OverflowError: #will this work???
			print "Overflow!!!"
			m_hesse = np.eye(HESSE_SIZE)
			delta_parameter = np.zeros([HESSE_SIZE, 1])
			pre_gradient = np.zeros([HESSE_SIZE, 1])
			restart()
			termination = 0
