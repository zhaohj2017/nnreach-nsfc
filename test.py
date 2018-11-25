import numpy as np
import gradient
import superpara
import math

def gentest():
	print '\n'
	print 'testing...'

	res = []

	testdata = np.zeros((superpara.INPUT_SIZE, 1))
	testdata[0, 0] = superpara.RANGE_Y[0] + superpara.EPSILON
	testdata[1, 0] = superpara.RANGE_T[1] - superpara.EPSILON
	testdata[2, 0] = 1


	while testdata[0, 0] <= superpara.RANGE_Y[1] - superpara.EPSILON:
		res.append(gradient.sol_candidate(testdata))
		testdata[0, 0] += superpara.MESH_SIZE_Y / 10.0

	#output
	#example: dy/dt = exp(y)
	print "test:", max(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[1] + superpara.EPSILON) - testdata[1, 0])
	print "test:", min(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[0] - superpara.EPSILON) - testdata[1, 0])

	"""
	#output
	#example: dy/dt = y
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1] - superpara.EPSILON) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0] + superpara.EPSILON) * np.exp(testdata[1, 0])
	"""