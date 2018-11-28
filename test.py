import numpy as np
import gradient
import superpara
import math
import matplotlib.pyplot as plt
import ann
import pipes


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
		testdata[0, 0] += superpara.MESH_SIZE_Y / 10.0


	
	#output**********************  dy / dt = exp(y)  *******************************************
	#example: dy / dt = exp(y)
	print "test:", max(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[1]) - testdata[1, 0])
	print "test:", min(res), "\treal:", - math.log(np.exp(- superpara.RANGE_Y[0]) - testdata[1, 0])
	print ""




	"""
	#output**********************  dy / dt = y  *******************************************
	#example: dy / dt = y
	print "test:", max(res), "\treal:", (superpara.RANGE_Y[1]) * np.exp(testdata[1, 0])
	print "test:", min(res), "\treal:", (superpara.RANGE_Y[0]) * np.exp(testdata[1, 0])
	print ""	
	"""








#*******************************************************************************************


def trajectory(input, step):
	hidden_input = ann.PIPES[step][0].dot(input)[:, 0]
	hidden_output = gradient.activation.activation_fun(hidden_input)
	nn_output = ann.PIPES[step][1].dot(hidden_output)
	return pipes.init(input, step) + (input[1, 0] - superpara.T_START) * nn_output



def reachplot(mesh_y, mesh_t):
	print '\n'
	print 'plotting...'

	time = []
	height = []
	bottom = []

	sample_y = np.arange(superpara.RANGE_Y[0], superpara.RANGE_Y[1] + mesh_y, mesh_y)
	

	for step in range(superpara.NUM_STEP):
		superpara.T_START = step * superpara.T_STEP
		if step == superpara.NUM_STEP - 1:
			t_step = np.arange(0, superpara.T_STEP + mesh_t, mesh_t) + superpara.T_START 
		else:
			t_step = np.arange(0, superpara.T_STEP, mesh_t) + superpara.T_START
		h_step = np.zeros(len(t_step))
		b_step = np.zeros(len(t_step))

		testdata = np.zeros((superpara.INPUT_SIZE, 1))
		testdata[2, 0] = 1

		for curr_t in t_step:
			res_y = []
			testdata[1, 0] = curr_t
			for curr_y in sample_y:
				testdata[0, 0] = curr_y
				res_y.append(trajectory(testdata, step))
			i = np.argwhere(t_step == curr_t) #get the index of curr_t in t_step
			h_step[i] = max(res_y) - min(res_y)		
			b_step[i] = min(res_y)

		time.extend(t_step)
		height.extend(h_step)
		bottom.extend(b_step)

	plt.bar(time, height, 0.0001, bottom)
	plt.show()
		