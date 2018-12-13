import numpy as np
import ode
import ann
import activation
import superpara
import test

#hidden layer input: return an array; x is a vector
def hidden_input(x):
	return ann.weight_matrix.dot(x)[:, 0] #matrix vector multiplication #[:, 0] is used to extract the array from the (n, 1) shape vector

#hidden layer output with activation functions and its primes: return an array
def hidden_output(x):
	return activation.activation_fun(hidden_input(x)) #vector activation function applied to hidden input

#return an array
def hidden_output_prime(x): 
	return activation.act_prime(hidden_input(x)) #matrix vector multiplication

#return an array
def hidden_output_prime_prime(x):
	return activation.act_prime_prime(hidden_input(x))  #matrix vector multiplication

#neural network output and its prime: return a scalar
def nn_output(x):
	return ann.weight_h_o.dot(hidden_output(x)) #inner product of two arrays

#return a scalar
def nn_output_prime(x):
	weight_product = ann.weight_h_o * ann.weight_t_h #element-wise product of two arrays
	return weight_product.dot(hidden_output_prime(x)) #inner product

#candidate solution: return a scalar (using the working weights, rather than the weights in pipes)
def sol_candidate(x, step):
	#first generate the initial condition for this step
	return test.init(x, step) + (x[1, 0] - step * superpara.T_STEP) * nn_output(x) 
			#the initial value is init(x) when t equals Time START: step * superpara.T_STEP

#derivative of solution candidate: dy / dt: return a scalar (using working weights, rather than weights in pipes)
def sol_dt(x, step):
	return nn_output(x) + (x[1, 0] - step * superpara.T_STEP) * nn_output_prime(x)

#partial derivatives of candidate solution w.r.t. the parameters: return an array
def temp_res1(x): # element-wise product of two arrays
	return ann.weight_h_o * hidden_output_prime(x)

def sol_dw_y_h(x, step): #dy / dwho: return an array
	return x[0, 0] * (x[1, 0] - step * superpara.T_STEP) * temp_res1(x)

def sol_dw_t_h(x, step): # return an array
	return x[1, 0] * (x[1, 0] - step * superpara.T_STEP) * temp_res1(x)

def sol_dw_b_h(x, step): # return an array
	return (x[1, 0] - step * superpara.T_STEP) * temp_res1(x)

def sol_dw_h_o(x, step): # return an array
	return (x[1, 0] - step * superpara.T_STEP) * hidden_output(x)

#second-order partial derivatives of candidate solution w.r.t. the parameters
def temp_res2(x): # return an array, element-wise product of two arrays
	return ann.weight_h_o * ann.weight_t_h * hidden_output_prime_prime(x)

def sol_dtw_y_h(x, step): #return an array
	return x[0, 0] * temp_res1(x) + x[0, 0] * (x[1, 0] - step * superpara.T_STEP) * temp_res2(x)

def sol_dtw_t_h(x, step): #return an array
	return (2.0 * x[1, 0] - step * superpara.T_STEP) * temp_res1(x) + x[1, 0] * (x[1, 0] - step * superpara.T_STEP) * temp_res2(x)

def sol_dtw_b_h(x, step): #return an array
	return temp_res1(x) + (x[1, 0] - step * superpara.T_STEP) * temp_res2(x)

def sol_dtw_h_o(x, step): #return an array
	return hidden_output(x) + (x[1, 0] - step * superpara.T_STEP) * hidden_output_prime(x) * ann.weight_t_h

#gradient of the cost function (error function)
def temp_res3(x, step): #return a scalar: the discrepency between computed and demanded derivatives
	return sol_dt(x, step) - ode.ode(sol_candidate(x, step), x[1, 0])

def gradient_dw_y_h(x, step): #return an array
	ode_dwyh = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_y_h(x, step) # scalar array product
	return temp_res3(x, step) * (sol_dtw_y_h(x, step) - ode_dwyh) # scalar array product

def gradient_dw_t_h(x, step): #return an array
	ode_dwth = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_t_h(x, step) # scalar array product	
	return temp_res3(x, step) * (sol_dtw_t_h(x, step) - ode_dwth) # scalar array product

def gradient_dw_b_h(x, step): #return an array
	ode_dw1h = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_b_h(x, step) # scalar array product
	return temp_res3(x, step) * (sol_dtw_b_h(x, step) - ode_dw1h) # scalar array product

def gradient_dw_h_o(x, step): #return an array
	ode_dwho = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_h_o(x, step) # scalar array product	
	return temp_res3(x, step) * (sol_dtw_h_o(x, step) - ode_dwho) # scalar array product
