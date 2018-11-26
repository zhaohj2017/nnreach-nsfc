import numpy as np
import gradient
import superpara
import math
import matplotlib.pyplot as plt

def chkprecision():
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

	"""
	#output**********************  dy / dt = exp(y)  *******************************************
	#example: dy / dt = exp(y)
	print "test:", max(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[1] + superpara.EPSILON) - testdata[1, 0])
	print "test:", min(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[0] - superpara.EPSILON) - testdata[1, 0])
	"""



	#output**********************  dy / dt = y  *******************************************
	#example: dy / dt = y
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1] - superpara.EPSILON) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0] + superpara.EPSILON) * np.exp(testdata[1, 0])

	



#plot the result
def reachplot(mesh_y, mesh_t):
	print '\n'
	print 'plotting...'

	sample_y = np.arange(superpara.RANGE_Y[0] + superpara.EPSILON, superpara.RANGE_Y[1] - superpara.EPSILON + mesh_y, mesh_y)
	sample_t = np.arange(superpara.RANGE_T[0] + superpara.EPSILON, superpara.RANGE_T[1] - superpara.EPSILON + mesh_t, mesh_t)

	height = np.zeros(len(sample_t))
	bottom = np.zeros(len(sample_t))

	testdata = np.zeros((superpara.INPUT_SIZE, 1))
	testdata[2, 0] = 1

	for curr_t in sample_t:
		res_y = []
		testdata[1, 0] = curr_t
		for curr_y in sample_y:
			testdata[0, 0] = curr_y
			res_y.append(gradient.sol_candidate(testdata))
		i = np.argwhere(sample_t == curr_t) #get the index of curr_t in sample_t
		bottom[i] = min(res_y)
		height[i] = max(res_y) - min(res_y)

	plt.bar(sample_t, height, 0.0001, bottom)
	plt.show()