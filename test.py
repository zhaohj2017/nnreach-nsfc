import numpy as np
import gradient
import superpara

def chkprecision(step):
	print '\n'
	print 'testing...'

	res = []

	testdata = np.zeros((superpara.INPUT_SIZE, 1))
	testdata[0, 0] = superpara.RANGE_Y[0]
	testdata[1, 0] = superpara.RANGE_T[1]
	testdata[2, 0] = 1

	while testdata[0, 0] <= superpara.RANGE_Y[1]:
		res.append(gradient.sol_candidate(testdata, step))
		testdata[0, 0] += superpara.MESH_SIZE_Y / superpara.TEST_RATE

	#example: dy / dt = y
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1]) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0]) * np.exp(testdata[1, 0])
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
