import numpy as np
import superpara
import activation
import ann

#the neural network weight computed for the ith (i >= 0) step is stored at index i in PIPES
PIPES = []

#after training, add the final value of weights to pipes
def addpipe():
	global pipes
	w_matrix = ann.weight_matrix.copy()
	w_h_o = ann.weight_h_o.copy()
	PIPES.append([w_matrix, w_h_o])

#print pipe
def printpipe():
	print "The pipes:"
	for weight in PIPES:
		print weight[0].T #transpose
		print weight[1]