import numpy as np
import superpara
import gradient
import ann
import random
import trainset
import activation

#global loop variables for bfgs quasi-newton
HESSE_SIZE = (superpara.INPUT_SIZE + 1) * superpara.NUM_HIDDEN
m_hesse = np.eye(HESSE_SIZE)
delta_weight = np.zeros([HESSE_SIZE, 1]) #store the difference between the current and previous points
pre_gradient = np.zeros([HESSE_SIZE, 1]) #store the gradient of the previous point
#errors between two epochs
error_pre = 0 #error of the previous epoch
error_curr = 0 #error of the current epoch
error_delta = 0 #difference between the error of the current and previous epochs


def error(w_matrix, w_ho, input, step): #square of the difference between derivatives: cost function, square error
	return np.square(gradient.temp_res3(w_matrix, w_ho, input, step))

#regularization error term
def reguerror(w_matrix, w_ho):
	w_yh = w_matrix[:, 0]   
	w_th = w_matrix[:, -2]  
	w_bh = w_matrix[:, -1]  #do we need to regularize the bias?
	#return superpara.REGU_FACTOR * (sum(np.square(w_yh)) + sum(np.square(w_th)) + sum(np.square(w_bh)) + sum(np.square(w_ho))) / 2.0
			 #do we need to regularize the bias?
	#return superpara.REGU_FACTOR * (sum(np.square(w_yh)) + sum(np.square(w_th)) + sum(np.square(w_ho))) / 2.0
			 #do not regularize the bias
	return np.sqrt((sum(np.square(w_yh)) + sum(np.square(w_th)) + sum(np.square(w_bh)) + sum(np.square(w_ho))) / (len(w_yh) + len(w_th) + len(w_bh) + len(w_ho)))
			 #do not regularize the bias

#gradient of regularization error term
def regugradient(w_matrix, w_ho):
	w_yh = w_matrix[:, 0]   
	w_th = w_matrix[:, -2]  
	w_bh = w_matrix[:, -1]  #do we need to regularize the bias?
	regugrad1 = np.append(w_yh, w_th)
	#regugrad2 = np.append(regugrad1, w_bh)
	regugrad2 = np.append(regugrad1, np.zeros(len(w_bh))) # do not regularize bias
	regugrad3 = np.append(regugrad2, w_ho)
	#reshape into a column vector
	regugrad = regugrad3.reshape(regugrad3.size, 1)
	#regularization factor
	return superpara.REGU_FACTOR * regugrad

def restarterror():
	#errors between two epochs
	global error_pre
	global error_curr
	global error_delta
	error_pre = 0
	error_curr = 0
	error_delta = 0

def restarthesse(): #reset variables and restart from epoch 0
	#loop variables for bfgs quasi-newton 
	global HESSE_SIZE
	global m_hesse
	global delta_weight
	global pre_gradient
	#update global varialbes
	m_hesse = np.eye(HESSE_SIZE)
	delta_weight = np.zeros([HESSE_SIZE, 1])
	pre_gradient = np.zeros([HESSE_SIZE, 1])

def restartweight(): #reset variables and restart from epoch 0
	#reset weights
	#rand_sigma = 1.0 / np.sqrt((superpara.INPUT_SIZE + 1) * superpara.NUM_HIDDEN)
	rand_sigma = 1.0
	rand_mu = 0
	ann.weight_matrix = rand_sigma * np.random.randn(superpara.NUM_HIDDEN, superpara.INPUT_SIZE) + rand_mu
	ann.weight_y_h = ann.weight_matrix[:, 0]    			#array (or matrix ?)
	ann.weight_t_h = ann.weight_matrix[:, -2]    			#array
	ann.weight_b_h = ann.weight_matrix[:, -1]    			#array
	ann.weight_h_o = np.random.randn(superpara.NUM_HIDDEN)	#array
	#ann.outweight()

def restart():
	restartweight()
	restarterror()
	restarthesse()

def success(curr_error, delta_error): #revise if needed
	if curr_error < 1e-4: #error < 0.0001
		print "Success! Stop!"
		return 1

def curr_error_grad(w_matrix, w_ho, batchset, step):
	#sum of gradient initialization for this mini batch
	w_yh = w_matrix[:, 0]    			#array (or matrix ?)
	w_th = w_matrix[:, -2]    			#array
	w_bh = w_matrix[:, -1]    			#array

	#initialization of sum of gradients and error
	sum_grad_wyh = np.zeros(len(w_yh))
	sum_grad_wth = np.zeros(len(w_th))
	sum_grad_wbh = np.zeros(len(w_bh))
	sum_grad_who = np.zeros(len(w_ho))
	#error for this mini batch
	error_batch = 0

	for inputdata in batchset:
		sum_grad_wyh += gradient.gradient_dw_y_h(w_matrix, w_ho, inputdata, step)
		sum_grad_wth += gradient.gradient_dw_t_h(w_matrix, w_ho, inputdata, step)
		sum_grad_wbh += gradient.gradient_dw_b_h(w_matrix, w_ho, inputdata, step)
		sum_grad_who += gradient.gradient_dw_h_o(w_matrix, w_ho, inputdata, step)
		#update error of this mini batch
		error_batch += error(w_matrix, w_ho, inputdata, step)
		#print "\terror_batch", error_batch

	#compute the average gradient for this mini batch, it's an array
	avg_gradient = np.append(sum_grad_wyh / len(batchset), sum_grad_wth / len(batchset))
	avg_gradient = np.append(avg_gradient, sum_grad_wbh / len(batchset))
	avg_gradient = np.append(avg_gradient, sum_grad_who / len(batchset))
	#reshape into a column vector
	curr_gradient = avg_gradient.reshape(avg_gradient.size, 1)
	#add gradient of regularization term
	curr_gradient = curr_gradient + regugradient(w_matrix, w_ho)

	return [error_batch, curr_gradient]

def updatew(alpha, direction): #direction is a column vector
	newmatrix = np.zeros(ann.weight_matrix.shape)
	newmatrix = newmatrix + (alpha * direction[:-superpara.NUM_HIDDEN, 0].reshape(newmatrix.T.shape)).T #cut the last num_hidden in direction
	newmatrix = ann.weight_matrix + newmatrix #update upon current weight

	newho = np.zeros(ann.weight_h_o.shape)
	newho = newho + alpha * direction[-superpara.NUM_HIDDEN:, 0]
	newho = newho + ann.weight_h_o #update upon current weight

	return [newmatrix, newho]

def linsearch(batchset, step, direction, curr_error, curr_gradient):
	"""
	#alpha1 = 0 
	#alpha2 = 1.0
	alpha = 1.0
	phi1 = curr_error / 2.0 / len(batchset) + reguerror(ann.weight_matrix, ann.weight_h_o) 
		# the cost function is the half of the sum of squares
		# plus the cost of the regularization term
	phi1_prime = curr_gradient[:, 0].dot(direction[:, 0]) #transform column vector into array and then take inner product
	c1 = 0.1
	#c2 = 0.4 #the smalller, the precise, the difficult
	
	flag1 = False
	#flag2 = False

	#simplified criteria
	while not(flag1) and alpha >= 1e-2:
		#update parameters
		new_matrix_ho = updatew(alpha, direction)
		new_matrix = new_matrix_ho[0]
		new_ho = new_matrix_ho[1]

		#new error and new gradient
		new_error_gradient = curr_error_grad(new_matrix, new_ho, batchset, step)
		new_error = new_error_gradient[0] / 2.0 / len(batchset) + reguerror(new_matrix, new_ho)
			# the cost function is the half of the sum of squares
			# plus the cost of the regularization term
		#new_gradient = new_error_gradient[1]

		#test condition 1 of Wolfe conditions
		flag1 = new_error <= phi1 + c1 * alpha * phi1_prime

		#binary search
		if not(flag1):
			alpha = alpha / 2.0

		return alpha
	"""

	#the above the the simplified criteria for linear search
	#the following conditions are Wolfe-Powell conditions for linear search

	"""
	itr = 0
	while not(flag1 and flag2) and itr < 5:
		#update parameters
		new_matrix_ho = updatew(alpha, direction)
		new_matrix = new_matrix_ho[0]
		new_ho = new_matrix_ho[1]

		#new error and new gradient
		new_error_gradient = curr_error_grad(new_matrix, new_ho, batchset, step)
		new_error = new_error_gradient[0] / 2.0 / len(batchset) + reguerror(new_matrix, new_ho)
			# the cost function is the half of the sum of squares
			# plus the cost of the regularization term
		new_gradient = new_error_gradient[1]

		#test condition 1 of Wolfe conditions
		flag1 = new_error <= phi1 + c1 * alpha * phi1_prime
		flag2 = new_gradient[:, 0].dot(direction[:, 0]) >= c2 * phi1_prime

		#use interpolation or binary search to update intervals

		# #interpolation
		# if not(flag1):
		# 	alpha_bar = alpha1 + (alpha - alpha1) / 2.0 / (1.0 + (phi1 - new_error) / (alpha - alpha1) / phi1_prime)
		# 	alpha2 = alpha
		# 	alpha = alpha_bar
		# elif not(flag2):
		# 	new_phi1_prime = new_gradient[:, 0].dot(direction[:, 0])
		# 	#different interpolation methods
		# 	#alpha_bar = alpha + (alpha - alpha1) * new_phi1_prime / (phi1_prime - new_phi1_prime)
		# 	alpha_bar = alpha1 + (alpha - alpha1) / 2.0 / (1.0 + (phi1 - new_error) / (alpha - alpha1) / phi1_prime)
		# 	alpha1 = alpha
		# 	alpha = alpha_bar
		# 	phi1_prime = new_phi1_prime
		# 	phi1 = new_error
		# else:
		# 	pass

		#binary search instead of interpolation
		if not(flag1):
			alpha2 = alpha
			alpha = (alpha1 + alpha2) / 2.0
		elif not(flag2):
			alpha1 = alpha
			alpha = (alpha1 + alpha2) / 2.0
		else:
			pass

		itr += 1

	return alpha
	"""
	return superpara.LEARN_RATE #the constant learning rate, superpara.LEARN_RATE


def bfgs(batchset, step):
	#global variables for bfgs quasi-newton 
	global HESSE_SIZE
	global m_hesse
	global delta_weight
	global pre_gradient

	#compute current batch error and batch gradient
	curr_error_gradient = curr_error_grad(ann.weight_matrix, ann.weight_h_o, batchset, step)
	error_batch = curr_error_gradient[0]
	curr_gradient = curr_error_gradient[1]

	#update delta gradient
	delta_gradient = curr_gradient - pre_gradient
	#update pre gradient
	pre_gradient = curr_gradient

	#update rho #initially delta_weight = 0, so rho = 0
	if delta_gradient[:, 0].dot(delta_weight[:, 0]) == 0: #exclude the case that the demoninator is zero
		rho = 0
	else:
		rho = 1.0 / delta_gradient[:, 0].dot(delta_weight[:, 0]) #exclude the case that the demoninator is zero

	#first update hesse matrix (for the initial loop, delta_weight = 0, so m_hesse is identity)
	hesse_temp1 = np.eye(HESSE_SIZE) - rho * np.dot(delta_weight, delta_gradient.T)
	hesse_temp2 = np.dot(hesse_temp1, m_hesse) #H_k
	hesse_temp3 = np.eye(HESSE_SIZE) - rho * np.dot(delta_gradient, delta_weight.T)
	hesse_temp4 = np.dot(hesse_temp2, hesse_temp3)
	m_hesse = hesse_temp4 + rho * np.dot(delta_weight, delta_weight.T) #initially, delta_weight is a zero vector

	#then update direction: a column vector, negtive gradient direction (minus)
	direction = - m_hesse.dot(curr_gradient) #for the first iteration, direction equals average gradient: SGD
	#then update learn rate
	rate = linsearch(batchset, step, direction, error_batch, curr_gradient)
	#update delta parameter: using minus rate
	delta_weight = rate * direction #update delta weight

	#update velocity
	ann.velocity = superpara.MOMENTUM_MU * ann.velocity + delta_weight

	#then update weights
	ann.weight_y_h += ann.velocity[0 : superpara.NUM_HIDDEN, 0] #extract
	ann.weight_t_h += ann.velocity[superpara.NUM_HIDDEN : superpara.NUM_HIDDEN * 2, 0]
	ann.weight_b_h += ann.velocity[superpara.NUM_HIDDEN * 2 : superpara.NUM_HIDDEN * 3, 0]
	ann.weight_h_o += ann.velocity[superpara.NUM_HIDDEN * 3 : direction.size, 0]

	#return the error of this batch after a number of bfgs iterations on this minibatch
	return [error_batch, rate] #return for output

def quasinewton(dataset, step):
	#errors between two epochs
	global error_pre
	global error_curr
	global error_delta

	#the number of mini batches for each epoch
	superpara.BATCH_NUM = len(dataset) / superpara.BATCH_SIZE
	if len(dataset) % superpara.BATCH_SIZE != 0: #the remaining data, not a whole batch
		superpara.BATCH_NUM += 1
	
	#start epoch iterations
	for epoch in range(superpara.EPOCHS):
		#shuffle training data 
		random.shuffle(dataset)

		#error of current epoch
		error_epoch = 0
		if superpara.PRINT_MINI == 1:
			print "epoch:", epoch

		#start batch traversal for this epoch
		for currbatch in range(superpara.BATCH_NUM):		
			#print "\tmini_batch:", currbatch
			#data set of this mini batch
			batchset = dataset[currbatch * superpara.BATCH_SIZE : (currbatch + 1) * superpara.BATCH_SIZE]

			#start bfgs iterations for current minibatch
			for iteration in range(superpara.BFGS_BATCH_ITR_NUM):
				#adjust learning rate
				superpara.LEARN_RATE = superpara.ALPHA / (1.0 + superpara.BETA * np.power(iteration, superpara.GAMMA))
				error_batch_curr_rate = bfgs(batchset, step)
				error_batch = error_batch_curr_rate[0]
				curr_rate = error_batch_curr_rate[1]
				if superpara.PRINT_MINI == 1:
					#print "\tbfgs iteration for this mini batch:", iteration, "error_batch average:", np.sqrt(error_batch / len(batchset)), "error_regu", reguerror(ann.weight_matrix, ann.weight_h_o), "rate:", curr_rate
					print "\tbfgs iteration for this mini batch:", iteration, "error_batch average:", np.sqrt(error_batch / len(batchset)), "rate:", curr_rate

			#reset hesse matrix for the next current batch
			#restart bfgs for a new minibatch
			restarthesse()

			#update epoch error using error of this mini batch
			error_epoch += error_batch
		
		#update errors between two adjacent epochs
		error_pre = error_curr
		error_curr = np.sqrt(error_epoch / len(dataset))	#average error, mean and squared
		error_delta = np.abs(error_curr - error_pre)
		#output
		print "epoch:", epoch, "error_curr:", error_curr, "error_pre:", error_pre, "error_delta:", error_delta, "rate:", superpara.LEARN_RATE
		
		if success(error_curr, error_delta) == 1: #early stop
			return 1 #early return

	restarterror()
	return 1 #all epochs finished, return 1, so the loop in itrdescent terminates

def itrdescent(dataset, step):
	termination = 0
	while termination == 0:
		try:
			termination = quasinewton(dataset, step)
		except(OverflowError, RuntimeWarning): #why do not work
			print "Run-time error!!!"
			restart() #update global variables
			termination = 0 #restart from epoch 0