import numpy as np
import gradient
import superpara

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
		testdata[0, 0] += 0.01

	#output
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1] - superpara.EPSILON) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0] + superpara.EPSILON) * np.exp(testdata[1, 0])