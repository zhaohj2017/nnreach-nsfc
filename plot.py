import numpy as np
import ann
import gradient
import pipes
import superpara
import matplotlib.pyplot as plt
import test

#plot the reachable set
def horiplot():
	print '\n'
	print 'plotting horizontally ...'

	#for each step
	for step in range(superpara.NUM_STEP):
		enlarge_y = step * superpara.ENLARGE_Y
		y_start = superpara.RANGE_Y[0] - enlarge_y
		y_end = superpara.RANGE_Y[1] + enlarge_y
		sample_y = np.arange(y_start, y_end + superpara.PLOT_MESH_Y * 0.99, superpara.PLOT_MESH_Y)
		
		t_start = step * superpara.T_STEP
		if (step + 1) * superpara.T_STEP > superpara.RANGE_T[1]:
			t_end = superpara.RANGE_T[1]
		else:
			t_end = (step + 1) * superpara.T_STEP
		sample_t = np.arange(t_start, t_end + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T) 

		trace_sy = np.zeros(len(sample_t))
		inputdata = np.zeros((superpara.INPUT_SIZE, 1))
		inputdata[2, 0] = 1
		
		for sy in sample_y:
			for i in range(len(sample_t)):
				inputdata[0, 0] = sy
				inputdata[1, 0] = sample_t[i]
				trace_sy[i] = test.trajectory(inputdata, step)
			plt.plot(sample_t, trace_sy, color = 'b', linestyle = '-')

	#working
	#plot the real upper and lower bounds for the example: dy / dt = exp(y)
	t = np.arange(0, superpara.LENGTH_T + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T)
	ytop = - np.log(np.exp(- superpara.RANGE_Y[1]) - t)
	ybtm = - np.log(np.exp(- superpara.RANGE_Y[0]) - t)
	plt.plot(t, ytop, color = 'r', linestyle = '-')
	plt.plot(t, ybtm, color = 'r', linestyle = '-')

	"""
	#plot the real upper and lower bounds for the example: dy / dt = y
	t = np.arange(0, superpara.LENGTH_T + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T)
	ytop = (superpara.RANGE_Y[1]) * np.exp(t)
	ybtm = (superpara.RANGE_Y[0]) * np.exp(t)
	plt.plot(t, ytop, color = 'r', linestyle = '-')
	plt.plot(t, ybtm, color = 'r', linestyle = '-')
	"""

	"""
	#plot the real upper and lower bounds for the example: dy / dt = exp(y)
	t = np.arange(0, superpara.LENGTH_T + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T)
	ytop = - np.log(np.exp(- superpara.RANGE_Y[1]) - t)
	ybtm = - np.log(np.exp(- superpara.RANGE_Y[0]) - t)
	plt.plot(t, ytop, color = 'r', linestyle = '-')
	plt.plot(t, ybtm, color = 'r', linestyle = '-')
	"""

	"""
	#plot the real upper and lower bounds for the example: dy / dt = 2 * t
	t = np.arange(0, superpara.LENGTH_T + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T)
	ytop = t * t + superpara.RANGE_Y[1]
	ybtm = t * t + superpara.RANGE_Y[0]
	plt.plot(t, ytop, color = 'r', linestyle = '-')
	plt.plot(t, ybtm, color = 'r', linestyle = '-')
	"""

	#show the plots
	plt.show()


def vertiplot():
	print '\n'
	print 'plotting vertically ...'

	time = []
	height = []
	bottom = []
	top = []
	
	for step in range(superpara.NUM_STEP):
		#sampling y
		enlarge_y = step * superpara.ENLARGE_Y
		y_start = superpara.RANGE_Y[0] - enlarge_y
		y_end = superpara.RANGE_Y[1] + enlarge_y
		sample_y = np.arange(y_start, y_end + superpara.PLOT_MESH_Y * 0.99, superpara.PLOT_MESH_Y)

		#sampling t
		t_start = step * superpara.T_STEP
		if (step + 1) * superpara.T_STEP > superpara.RANGE_T[1]:
			t_end = superpara.RANGE_T[1]
		else:
			t_end = (step + 1) * superpara.T_STEP
		sample_t = np.arange(t_start, t_end + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T) 

		h_step = np.zeros(len(sample_t))
		b_step = np.zeros(len(sample_t))
		tp_step = np.zeros(len(sample_t))

		testdata = np.zeros((superpara.INPUT_SIZE, 1))
		testdata[2, 0] = 1

		for curr_t in sample_t:
			res_y = []
			testdata[1, 0] = curr_t
			for curr_y in sample_y:
				testdata[0, 0] = curr_y
				res_y.append(test.trajectory(testdata, step))
			i = np.argwhere(sample_t == curr_t) #get the index of curr_t in t_step
			h_step[i] = max(res_y) - min(res_y)		
			b_step[i] = min(res_y)
			tp_step[i] = max(res_y)

		time.extend(sample_t)
		height.extend(h_step)
		bottom.extend(b_step)
		top.extend(tp_step)
	
	#plot reachable set
	plt.bar(time, np.array(height), 0.0001, bottom, facecolor = 'g', edgecolor = 'g')

	#plot the upper and lower bounds for the example: dy / dt = exp(y)
	t = np.arange(0, superpara.LENGTH_T + superpara.PLOT_MESH_T * 0.99, superpara.PLOT_MESH_T)
	ytop = - np.log(np.exp(- superpara.RANGE_Y[1]) - t)
	ybtm = - np.log(np.exp(- superpara.RANGE_Y[0]) - t)
	#the real upper and lower bounds
	plt.plot(t, ytop, color = 'r', linestyle = '-')
	plt.plot(t, ybtm, color = 'r', linestyle = '-')
	
	#show the plots
	plt.show()
