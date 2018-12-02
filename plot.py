import numpy as np
import ann
import gradient
import pipes
import superpara
import matplotlib.pyplot as plt


def trajectory(input, step):
	hidden_input = ann.PIPES[step][0].dot(input)[:, 0]
	hidden_output = gradient.activation.activation_fun(hidden_input)
	nn_output = ann.PIPES[step][1].dot(hidden_output)
	return pipes.init(input, step) + (input[1, 0] - step * superpara.T_STEP) * nn_output

def reachplot(mesh_y, mesh_t):
	print '\n'
	print 'plotting...'

	time = []
	height = []
	bottom = []
	top = []

	sample_y = np.arange(superpara.RANGE_Y[0], superpara.RANGE_Y[1] + mesh_y, mesh_y)
	
	for step in range(superpara.NUM_STEP):
		if step == superpara.NUM_STEP - 1:
			t_step = np.arange(0, superpara.T_STEP + mesh_t, mesh_t) + step * superpara.T_STEP 
		else:
			t_step = np.arange(0, superpara.T_STEP, mesh_t) + step * superpara.T_STEP
		h_step = np.zeros(len(t_step))
		b_step = np.zeros(len(t_step))
		tp_step = np.zeros(len(t_step))

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
			tp_step[i] = max(res_y)

		time.extend(t_step)
		height.extend(h_step)
		bottom.extend(b_step)
		top.extend(tp_step)

	#plot reachable set
	#plt.bar(time, np.array(height) + 0.0001, 0.0001, bottom)

	#plot the upper and lower bounds for the example: dy / dt = exp(y)
	t = np.arange(0, superpara.T_STEP * superpara.NUM_STEP + superpara.PLOT_MESH_T, superpara.PLOT_MESH_T)
	ytop = - np.log(np.exp(- superpara.RANGE_Y[1]) - t)
	ybtm = - np.log(np.exp(- superpara.RANGE_Y[0]) - t)

	#the real upper and lower bounds
	plt.plot(t, ytop)
	plt.plot(t, ybtm)
	#the upper and lower bounds given by neural network
	plt.plot(t, bottom)
	plt.plot(t, top)

	#show the plots
	plt.show()
		