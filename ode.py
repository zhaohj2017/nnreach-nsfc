import numpy as np

"""
#ode
def ode(y, t):
	return y # dy/dt = y

#ode_derivative
def ode_dy(y, t):
	return 1 # dy/dt = y
"""

def ode(y, t):
	return np.exp(y) # dy/dt = exp(y)

def ode_dy(y, t):
	return np.exp(y) # dy/dt = exp(y)