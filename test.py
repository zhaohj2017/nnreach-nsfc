import numpy as np
import gradient
import superpara
import ann
import pipes
import activation

#when calling methods in test.py, the pipes has already be generated
#so we call use pipes to test inputs

#return the output of step i: a scalar
#learned trajectory, different from sol_candidate in gradient.py
def trajectory(input, step):
	#first generate the initial condition for this step
	return gradient.sol_candidate(pipes.PIPES[step][0], pipes.PIPES[step][1], input, step) 
			#the initial value is init(x) when t equals Time START: step * superpara.T_STEP

#check the precision using the computed pipes and the closed form solutions
def chkprecision(step):
	print '\n'
	print 'testing...'

	res = []

	testdata = np.zeros((superpara.INPUT_SIZE, 1))
	testdata[0, 0] = superpara.RANGE_Y[0] #the left end of y
	if (step + 1) * superpara.T_STEP > superpara.RANGE_T[1]:
		testdata[1, 0] = superpara.RANGE_T[1]
	else:
		testdata[1, 0] = (step + 1) * superpara.T_STEP #use the right end of t of this step as the terminal t
	testdata[2, 0] = 1

	while testdata[0, 0] <= superpara.RANGE_Y[1] + 1e-3 * superpara.PLOT_MESH_Y: #until the right end of y
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

	"""
	#output**********************  dy / dt = t *******************************************
	#example: dy / dt = t
	print "test:", max(res), "\treal:", testdata[1, 0] * testdata[1, 0] + superpara.RANGE_Y[1]
	print "test:", min(res), "\treal:", testdata[1, 0] * testdata[1, 0] + superpara.RANGE_Y[0]
	print ""
	"""

