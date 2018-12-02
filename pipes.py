import numpy as np
import superpara
import activation
import ann


def        (input, step):
	output = input[0, 0]
	tempinput = input.copy()
	for i in range(step): # range(0) is empty, so output = input !!!!
		tempinput[1, 0] = superpara.T_STEP * (i + 1)
		hidden_input = ann.PIPES[i][0].dot(tempinput)[:, 0]
		hidden_output = activation.activation_fun(hidden_input)
		nn_output = ann.PIPES[i][1].dot(hidden_output)
		output += superpara.T_STEP * nn_output
	return output

def addpipe():
	w_matrix = ann.weight_matrix.copy()
	w_h_o = ann.weight_h_o.copy()
	ann.PIPES.append([w_matrix, w_h_o])

def printpipe():
	print "The pipes:"
	for weight in ann.PIPES:
		print weight[0].T
		print weight[1]