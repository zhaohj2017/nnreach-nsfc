import numpy as np
import ode
import ann
import activation
import superpara
import test
import pipes

#hidden layer input: return an array; x is a vector
def hidden_input(w_matrix, input):
	return w_matrix.dot(input)[:, 0] #matrix vector multiplication #[:, 0] is used to extract the array from the (n, 1) shape vector

#hidden layer output with activation functions and its primes: return an array
def hidden_output(w_matrix, input):
	return activation.activation_fun(hidden_input(w_matrix, input)) #vector activation function applied to hidden input

#return an array
def hidden_output_prime(w_matrix, input): 
	return activation.act_prime(hidden_input(w_matrix, input)) #matrix vector multiplication

#return an array
def hidden_output_prime_prime(w_matrix, input):
	return activation.act_prime_prime(hidden_input(w_matrix, input))  #matrix vector multiplication

#neural network output and its prime: return a scalar
def nn_output(w_matrix, w_ho, input):
	return w_ho.dot(hidden_output(w_matrix, input)) #inner product of two arrays

#return a scalar
def nn_output_prime(w_matrix, w_ho, input):
	weight_product = w_ho * w_matrix[:, -2] #element-wise product of two arrays
	return weight_product.dot(hidden_output_prime(w_matrix, input)) #inner product

#generate the initial condition for step i: a scalar; how about higher dimension?
def init(input, step):
	output = input[0, 0] # a scalar
	tempinput = input.copy() # an array
	for i in range(step): # range(0) is empty, so output = input !!!!
		tempinput[1, 0] = superpara.T_STEP * (i + 1)
		output += superpara.T_STEP * nn_output(pipes.PIPES[i][0], pipes.PIPES[i][1], tempinput)
	return output

#candidate solution: return a scalar (using the working weights, rather than the weights in pipes)
def sol_candidate(w_weight, w_ho, input, step):
	#first generate the initial condition for this step
	return init(input, step) + (input[1, 0] - step * superpara.T_STEP) * nn_output(w_weight, w_ho, input) 
			#the initial value is init(x) when t equals Time START: step * superpara.T_STEP

#derivative of solution candidate: dy / dt: return a scalar (using working weights, rather than weights in pipes)
def sol_dt(w_matrix, w_ho, input, step):
	return nn_output(w_matrix, w_ho, input) + (input[1, 0] - step * superpara.T_STEP) * nn_output_prime(w_matrix, w_ho, input)

#partial derivatives of candidate solution w.r.t. the parameters: return an array
def temp_res1(w_matrix, w_ho, input): # element-wise product of two arrays
	return w_ho * hidden_output_prime(w_matrix, input)

def sol_dw_y_h(w_matrix, w_ho, input, step): #dy / dwho: return an array
	return input[0, 0] * (input[1, 0] - step * superpara.T_STEP) * temp_res1(w_matrix, w_ho, input)

def sol_dw_t_h(w_matrix, w_ho, input, step): # return an array
	return input[1, 0] * (input[1, 0] - step * superpara.T_STEP) * temp_res1(w_matrix, w_ho, input)

def sol_dw_b_h(w_matrix, w_ho, input, step): # return an array
	return (input[1, 0] - step * superpara.T_STEP) * temp_res1(w_matrix, w_ho, input)

def sol_dw_h_o(w_matrix, input, step): # return an array
	return (input[1, 0] - step * superpara.T_STEP) * hidden_output(w_matrix, input)

#second-order partial derivatives of candidate solution w.r.t. the parameters
def temp_res2(w_matrix, w_ho, input): # return an array, element-wise product of two arrays
	return w_ho * w_matrix[:, -2] * hidden_output_prime_prime(w_matrix, input)

def sol_dtw_y_h(w_matrix, w_ho, input, step): #return an array
	return input[0, 0] * temp_res1(w_matrix, w_ho, input) + input[0, 0] * (input[1, 0] - step * superpara.T_STEP) * temp_res2(w_matrix, w_ho, input)

def sol_dtw_t_h(w_matrix, w_ho, input, step): #return an array
	return (2.0 * input[1, 0] - step * superpara.T_STEP) * temp_res1(w_matrix, w_ho, input) + input[1, 0] * (input[1, 0] - step * superpara.T_STEP) * temp_res2(w_matrix, w_ho, input)

def sol_dtw_b_h(w_matrix, w_ho, input, step): #return an array
	return temp_res1(w_matrix, w_ho, input) + (input[1, 0] - step * superpara.T_STEP) * temp_res2(w_matrix, w_ho, input)

def sol_dtw_h_o(w_matrix, w_ho, input, step): #return an array
	return hidden_output(w_matrix, input) + (input[1, 0] - step * superpara.T_STEP) * hidden_output_prime(w_matrix, input) * w_matrix[:, -2]

#the error function: not squared
def temp_res3(w_matrix, w_ho, input, step): #return a scalar: the discrepency between computed and demanded derivatives
	return sol_dt(w_matrix, w_ho, input, step) - ode.ode(sol_candidate(w_matrix, w_ho, input, step), input[1, 0])

def gradient_dw_y_h(w_matrix, w_ho, input, step): #return an array
	ode_dwyh = ode.ode_dy(sol_candidate(w_matrix, w_ho, input, step), input[1, 0]) * sol_dw_y_h(w_matrix, w_ho, input, step) # scalar array product
	return temp_res3(w_matrix, w_ho, input, step) * (sol_dtw_y_h(w_matrix, w_ho, input, step) - ode_dwyh) # scalar array product

def gradient_dw_t_h(w_matrix, w_ho, input, step): #return an array
	ode_dwth = ode.ode_dy(sol_candidate(w_matrix, w_ho, input, step), input[1, 0]) * sol_dw_t_h(w_matrix, w_ho, input, step) # scalar array product	
	return temp_res3(w_matrix, w_ho, input, step) * (sol_dtw_t_h(w_matrix, w_ho, input, step) - ode_dwth) # scalar array product

def gradient_dw_b_h(w_matrix, w_ho, input, step): #return an array
	ode_dw1h = ode.ode_dy(sol_candidate(w_matrix, w_ho, input, step), input[1, 0]) * sol_dw_b_h(w_matrix, w_ho, input, step) # scalar array product
	return temp_res3(w_matrix, w_ho, input, step) * (sol_dtw_b_h(w_matrix, w_ho, input, step) - ode_dw1h) # scalar array product

def gradient_dw_h_o(w_matrix, w_ho, input, step): #return an array
	ode_dwho = ode.ode_dy(sol_candidate(w_matrix, w_ho, input, step), input[1, 0]) * sol_dw_h_o(w_matrix, input, step) # scalar array product	
	return temp_res3(w_matrix, w_ho, input, step) * (sol_dtw_h_o(w_matrix, w_ho, input, step) - ode_dwho) # scalar array product
