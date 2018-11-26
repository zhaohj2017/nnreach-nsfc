import numpy as np
import ann


"""
#***************** dy / dt = exp(y) *******************
def ode(y, t):
	#print ann.weight_h_o
	#print y
	return np.exp(y) # dy / dt = exp(y)

def ode_dy(y, t):
	return np.exp(y) # dy / dy = exp(y)
"""




#***************** dy / dt = y *******************
#ode
def ode(y, t):
	return y # dy / dt = y

#ode_derivative
def ode_dy(y, t):
	return 1 # dy / dy = 1


