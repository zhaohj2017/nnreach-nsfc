import numpy as np
import ode
import ann
import activation
import superpara
import pipes

#hidden layer input
def hidden_input(x):
	return ann.weight_matrix.dot(x)[:, 0] #matrix vector multiplication #[:, 0] is used to extract the array from the (n, 1) shape vector

#hidden layer output with activation functions and its primes
def hidden_output(x):
	return activation.activation_fun(hidden_input(x)) #vector activation function applied to hidden input

def hidden_output_prime(x): 
	return activation.act_prime(hidden_input(x)) #matrix vector multiplication

def hidden_output_prime_prime(x):
	return activation.act_prime_prime(hidden_input(x))  #matrix vector multiplication

#neural network output and its prime
def nn_output(x):
	return ann.weight_h_o.dot(hidden_output(x)) #inner product of two arrays

def nn_output_prime(x):
	weight_product = ann.weight_h_o * ann.weight_t_h #element-wise product of two arrays
	return weight_product.dot(hidden_output_prime(x)) #inner product

#candidate solution
def sol_candidate(x, step):
	#first generate the initial condition for this step
	return pipes.init(x, step) + (x[1, 0] - superpara.T_START) * nn_output(x) #the initial value is init(x) when t equals T_START

#derivative of solution candidate: dy / dt
def sol_dt(x):
	return nn_output(x) + (x[1, 0] - superpara.T_START) * nn_output_prime(x)

#partial derivatives of candidate solution w.r.t. the parameters 
def temp_res1(x):
	return ann.weight_h_o * hidden_output_prime(x)

def sol_dw_y_h(x): #dy / dwho
	return x[0, 0] * (x[1, 0] - superpara.T_START) * temp_res1(x)

def sol_dw_t_h(x):
	return x[1, 0] * (x[1, 0] - superpara.T_START) * temp_res1(x)

def sol_dw_b_h(x):
	return (x[1, 0] - superpara.T_START) * temp_res1(x)

def sol_dw_h_o(x):
	return (x[1, 0] - superpara.T_START) * hidden_output(x)

#second-order partial derivatives of candidate solution w.r.t. the parameters
def temp_res2(x):
	return ann.weight_h_o * ann.weight_t_h * hidden_output_prime_prime(x)

def sol_dtw_y_h(x):
	return x[0, 0] * temp_res1(x) + x[0, 0] * (x[1, 0] - superpara.T_START) * temp_res2(x)

def sol_dtw_t_h(x):
	return (2.0 * x[1, 0] - superpara.T_START) * temp_res1(x) + x[1, 0] * (x[1, 0] - superpara.T_START) * temp_res2(x)

def sol_dtw_b_h(x):
	return temp_res1(x) + (x[1, 0] - superpara.T_START) * temp_res2(x)

def sol_dtw_h_o(x):
	return hidden_output(x) + (x[1, 0] - superpara.T_START) * hidden_output_prime(x) * ann.weight_t_h

#gradient of the cost function (error function)
def temp_res3(x, step):
	return sol_dt(x) - ode.ode(sol_candidate(x, step), x[1, 0])

def gradient_dw_y_h(x, step):
	ode_dwyh = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_y_h(x)	
	return temp_res3(x, step) * (sol_dtw_y_h(x) - ode_dwyh)

def gradient_dw_t_h(x, step):
	ode_dwth = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_t_h(x)	
	return temp_res3(x, step) * (sol_dtw_t_h(x) - ode_dwth)

def gradient_dw_b_h(x, step):
	ode_dw1h = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_b_h(x)	
	return temp_res3(x, step) * (sol_dtw_b_h(x) - ode_dw1h)

def gradient_dw_h_o(x, step):
	ode_dwho = ode.ode_dy(sol_candidate(x, step), x[1, 0]) * sol_dw_h_o(x)	
	return temp_res3(x, step) * (sol_dtw_h_o(x) - ode_dwho)
