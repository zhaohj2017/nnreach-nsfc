import numpy as np
import superpara
import activation
import ann

#after training, add the final value of weights to pipes
def addpipe():
	w_matrix = ann.weight_matrix.copy()
	w_h_o = ann.weight_h_o.copy()
	ann.PIPES.append([w_matrix, w_h_o])

#print pipe
def printpipe():
	print "The pipes:"
	for weight in ann.PIPES:
		print weight[0].T #transpose
		print weight[1]