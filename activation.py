import numpy as np


#activation prime
def act_prime(x):
	return activation_fun(x) * (1 - activation_fun(x))

#activation prime prime
def act_prime_prime(x):
	return act_prime(x) - 2 * activation_fun(x) * act_prime(x)





"""
#other activation functions?


"""


#sigmoid
def activation_fun(x): #act on each element in a matrix
	return 1.0 / (1.0 + np.exp(-x))



"""
#relu
def activation_fun(x): #act on each element in a matrix
	return (x > 0) * x

#activation prime
def act_prime(x):
	return (x > 0) * 1

#activation prime prime
def act_prime_prime(x):
	return np.zeros(x.shape)
"""