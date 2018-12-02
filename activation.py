import numpy as np

#working for dy / dt = y
#**************************************
#sigmoid
def activation_fun(x): #act on each element in a matrix
	return 1.0 / (1.0 + np.exp(-x))

#activation prime
def act_prime(x):
	return activation_fun(x) * (1.0 - activation_fun(x))

#activation prime prime
def act_prime_prime(x):
	return act_prime(x) - 2.0 * activation_fun(x) * act_prime(x)


"""
##****************** sigmoid function *****************
#sigmoid
def activation_fun(x): #act on each element in a matrix
	return 1.0 / (1.0 + np.exp(-x))

#activation prime
def act_prime(x):
	return activation_fun(x) * (1.0 - activation_fun(x))

#activation prime prime
def act_prime_prime(x):
	return act_prime(x) - 2.0 * activation_fun(x) * act_prime(x)
"""



"""
#******************* fast sigmoid function ********************
#fast sigmoid
#negtivity
#negtive: return 1; positive: return -1
def neg(x):
	return -((x > 0) * 2 - 1)

def activation_fun(x):
	return x / (1.0 + np.abs(x))

def act_prime(x):
	return 1.0 / np.square(1.0 + np.abs(x))

def act_prime_prime(x):
	return 2.0 * neg(x) / np.power(1.0 + np.abs(x), 3)
"""




"""
#******************* hyperbolic function ********************
#tanh
def activation_fun(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def act_prime(x):
	return 1.0 - np.square(activation_fun(x))

def act_prime_prime(x):
	return -2.0 * activation_fun(x) * act_prime(x)
"""




"""
#res: use small rate, converge quickly, but not stable
#******************** RELU function ********************
#relu
def activation_fun(x): #act on each element in a matrix
	return (x > 0) * x * 1.0
#activation prime
def act_prime(x):
	return (x > 0) * 1.0

#activation prime prime
def act_prime_prime(x):
	return np.zeros(x.shape)
"""