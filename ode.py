import numpy as np
import ann



#working
#***************** dy / dt = y *******************
#ode
# def ode(y, t):
# 	return y * y # dy / dt = y

# #ode_derivative
# def ode_dy(y, t):
# 	return 2 * y # dy / dy = 1
def ode(y, t):
	return np.exp(y) # dy / dt = y

#ode_derivative
def ode_dy(y, t):
	return np.exp(y)# dy / dy = 1


"""
#***************** dy / dt = y *******************
#ode
def ode(y, t):
	return y # dy / dt = y

#ode_derivative
def ode_dy(y, t):
	return 1 # dy / dy = 1
"""


"""
#***************** dy / dt = exp(y) *******************
def ode(y, t):
	return np.exp(y) # dy / dt = exp(y)

def ode_dy(y, t):
	return np.exp(y) # dy / dy = exp(y)
"""





