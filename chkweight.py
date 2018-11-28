import superpara
import ann

def outweight():

	print "The weight matrix"
	print ann.weight_matrix.T

	print "The weight w.r.t. each input state"
	print ann.weight_y_h
	print ann.weight_t_h
	print ann.weight_b_h

	print "The weight w.r.t. the output"
	print ann.weight_h_o

	print ""
	