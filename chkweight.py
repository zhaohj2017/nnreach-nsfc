import superpara
import ann

def outweight():

	"""
	print "The weight matrix"
	print ann.weight_matrix.T
	"""

	print "The weight w.r.t. each input state"
	print "weight-input-y-hidden"
	print ann.weight_y_h
	print "weight-input-t-hidden"
	print ann.weight_t_h
	print "bias"
	print ann.weight_b_h

	print "weight-hidden-output"
	print ann.weight_h_o

	print ""
	