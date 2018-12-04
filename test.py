import numpy as np
import gradient
import superpara
import ann
import pipes
import activation

#when calling methods in test.py, the pipes has already be generated
#so we call use pipes to test inputs

#generate the initial condition for step i: a scalar
def init(input, step):
	output = input[0, 0] # a scalar
	tempinput = input.copy() # an array
	for i in range(step): # range(0) is empty, so output = input !!!!
		tempinput[1, 0] = superpara.T_STEP * (i + 1)
		hidden_input = ann.PIPES[i][0].dot(tempinput)[:, 0]
		hidden_output = activation.activation_fun(hidden_input)
		nn_output = ann.PIPES[i][1].dot(hidden_output)
		output += superpara.T_STEP * nn_output
	return output

#return the output of step i: a scalar
def trajectory(input, step):
	hidden_input = ann.PIPES[step][0].dot(input)[:, 0] # an array: extract an array from a vector of shape (n, 1)
	hidden_output = gradient.activation.activation_fun(hidden_input) # an array
	nn_output = ann.PIPES[step][1].dot(hidden_output) # a scalar
	return init(input, step) + (input[1, 0] - step * superpara.T_STEP) * nn_output # a scalar

#check the precision using the computed pipes and the closed form solutions
def chkprecision(step):
	print '\n'
	print 'testing...'

	res = []

	testdata = np.zeros((superpara.INPUT_SIZE, 1))
	testdata[0, 0] = superpara.RANGE_Y[0] #the left end of y
	testdata[1, 0] = (step + 1) * superpara.T_STEP #use the right end of t of this step as the terminal t
	testdata[2, 0] = 1

	while testdata[0, 0] <= superpara.RANGE_Y[1]: #until the right end of y
		res.append(trajectory(testdata, step)) #call trajectory
		testdata[0, 0] += superpara.PLOT_MESH_Y # test points should be a lot more than training points

	#working...
	#output**********************  dy / dt = exp(y)  *******************************************
	#example: dy / dt = exp(y)
	print "test:", max(res), "\treal:", - np.log(np.exp(- superpara.RANGE_Y[1]) - testdata[1, 0])
	print "test:", min(res), "\treal:", - np.log(np.exp(- superpara.RANGE_Y[0]) - testdata[1, 0])
	print ""


	"""
	#output**********************  dy / dt = exp(y)  *******************************************
	#example: dy / dt = exp(y)
	print "test:", max(res), "\treal:", - np.log(np.exp(- superpara.RANGE_Y[1]) - testdata[1, 0])
	print "test:", min(res), "\treal:", - np.log(np.exp(- superpara.RANGE_Y[0]) - testdata[1, 0])
	print ""
	"""

	"""
	#output**********************  dy / dt = y  *******************************************
	#example: dy / dt = y
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1]) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0]) * np.exp(testdata[1, 0])
	print ""	
	"""
