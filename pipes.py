import numpy as np
import superpara
import activation
import ann


def hidden_input(x):
	return ann.weight_matrix.dot(x) #matrix vector multiplication

def hidden_output(x):
	return activation.activation_fun(hidden_input(x))[:, 0] 

def nn_output(x):
	return ann.weight_h_o.dot(hidden_output(x)) #inner product of two arrays



def geninit(input, i):
	output = input[0, 0]
	tempinput = np.zeros(input.shape)
	tempinput[0, 0] = input[0, 0]
	tempinput[2, 0] = 1
	for j in range(i):
		tempinput[1, 0] = superpara.T_STEP * j
		hidden_input = ann.PIPES[j][0].dot(tempinput)
		hidden_output = activation.activation_fun(hidden_input)
		nn_output = ann.PIPES[j][1].dot(hidden_output)
		output += tempinput[1, 0] * nn_output
	return output
	

def init(input):
	index = int(input[1, 0] / superpara.T_STEP) + 1
	return geninit(input, index)



