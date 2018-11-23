import numpy as np
import ode
import ann
import activation

#hidden layer input
def hidden_input(x):
	return ann.weight_matrix.dot(x)

#hidden layer output
def hidden_output(x):
	return activation.activation_fun(hidden_input(x))[:, 0] #vector activation function

def hidden_output_prime(x): 
	return activation.act_prime(hidden_input(x))[:, 0] #matrix vector multiplication

def hidden_output_prime_prime(x):
	return activation.act_prime_prime(hidden_input(x))[:, 0]  #matrix vector multiplication

#neural network output
def nn_output(x):
	return ann.weight_h_o.dot(hidden_output(x))

#solution candidate
def sol_candidate(x):
	return x[0, 0] + x[1, 0] * nn_output(x)

#derivative of solution candidate
def sol_dt(x):
	weight_product = ann.weight_h_o * ann.weight_t_h
	return nn_output(x) + x[1, 0] * weight_product.dot(hidden_output_prime(x))

#partial derivatives of candidate solution w.r.t. the parameters 
def temp_res1(x):
	return ann.weight_h_o * hidden_output_prime(x)

def sol_dw_y_h(x):
	return x[0, 0] * x[1, 0] * temp_res1(x)

def sol_dw_t_h(x):
	return x[1, 0] * x[1, 0] * temp_res1(x)

def sol_dw_b_h(x):
	return x[1, 0] * temp_res1(x)

def sol_dw_h_o(x):
	return x[1, 0] * hidden_output(x)

#second-order partial derivatives of candidate solution w.r.t. the parameters
def temp_res2(x):
	return ann.weight_h_o * ann.weight_t_h * hidden_output_prime_prime(x)

def sol_dtw_y_h(x):
	return x[0, 0] * temp_res1(x) + x[0, 0] * x[1, 0] * temp_res2(x)

def sol_dtw_t_h(x):
	return 2 * x[1, 0] * temp_res1(x) + x[1, 0] * x[1, 0] * temp_res2(x)

def sol_dtw_b_h(x):
	return temp_res1(x) + x[1, 0] * temp_res2(x)

def sol_dtw_h_o(x):
	return hidden_output(x) + x[1, 0] * hidden_output_prime(x) * ann.weight_t_h

#gradient of the cost function (error function)
def temp_res3(x):
	return sol_dt(x) - ode.ode(sol_candidate(x), x[1, 0])

def gradient_dw_y_h(x):
	ode_dwyh = ode.ode_dy(sol_candidate(x), x[1, 0]) * sol_dw_y_h(x)	
	return temp_res3(x) * (sol_dtw_y_h(x) - ode_dwyh)

def gradient_dw_t_h(x):
	ode_dwth = ode.ode_dy(sol_candidate(x), x[1, 0]) * sol_dw_t_h(x)	
	return temp_res3(x) * (sol_dtw_t_h(x) - ode_dwth)

def gradient_dw_b_h(x):
	ode_dw1h = ode.ode_dy(sol_candidate(x), x[1, 0]) * sol_dw_b_h(x)	
	return temp_res3(x) * (sol_dtw_b_h(x) - ode_dw1h)

def gradient_dw_h_o(x):
	ode_dwho = ode.ode_dy(sol_candidate(x), x[1, 0]) * sol_dw_h_o(x)	
	return temp_res3(x) * (sol_dtw_h_o(x) - ode_dwho)
