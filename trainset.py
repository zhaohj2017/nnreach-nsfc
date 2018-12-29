import superpara
import numpy as np

def gendata(step):
	datalist = []
	sample_y = []
	sample_t = []

	#very important!!!
	#update T-RANGE
	t_start = step * superpara.T_STEP - superpara.EPS_T
	if (step + 1) * superpara.T_STEP > superpara.RANGE_T[1]:
		t_end = superpara.RANGE_T[1] + superpara.EPS_T
	else:
		t_end = (step + 1) * superpara.T_STEP + superpara.EPS_T
	#sample point by meshing
	sample_t = np.arange(t_start, t_end + superpara.MESH_SIZE_T * 0.99, superpara.MESH_SIZE_T) 
		#plus superpara.MESH_SIZE_T * 0.99 to include the right end point

	enlarge_y = step * superpara.ENLARGE_Y
	y_start = superpara.RANGE_Y[0] - enlarge_y - superpara.EPS_Y
	y_end = superpara.RANGE_Y[1] + enlarge_y + superpara.EPS_Y
	sample_y = np.arange(y_start, y_end + superpara.MESH_SIZE_Y * 0.99, superpara.MESH_SIZE_Y)

	""" obsolete??? not sure!!!
	if len(sample_y) == 0: # a single point inital y
		sample_y = np.array([superpara.RANGE_Y[0]])
	"""
	
	#sample data generation by gridding
	for st in sample_t:
		for sy in sample_y:
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[0, 0] = sy
			data[1, 0] = st
			data[superpara.INPUT_SIZE - 1, 0] = 1
			datalist.append(data)
		#sample_y = np.fliplr([sample_y])[0] #reverse: not helpful? discard!

	
	
	"""
	#sample data generation by uniform distribution
	data_size = len(sample_t) * len(sample_y)
	for i in np.arange(data_size):
		data = np.zeros((superpara.INPUT_SIZE, 1))
		data[0, 0] = np.random.uniform(y_start, y_end)
		data[1, 0] = np.random.uniform(t_start, t_end)
		data[superpara.INPUT_SIZE - 1, 0] = 1
		datalist.append(data)

	corner1 = np.zeros((superpara.INPUT_SIZE, 1))
	corner1[0, 0] = superpara.RANGE_Y[0]
	corner1[1, 0] = superpara.RANGE_T[0]
	corner1[superpara.INPUT_SIZE - 1, 0] = 1

	corner2 = np.zeros((superpara.INPUT_SIZE, 1))
	corner2[0, 0] = superpara.RANGE_Y[0]
	corner2[1, 0] = superpara.RANGE_T[1]
	corner2[superpara.INPUT_SIZE - 1, 0] = 1

	corner3 = np.zeros((superpara.INPUT_SIZE, 1))
	corner3[0, 0] = superpara.RANGE_Y[1]
	corner3[1, 0] = superpara.RANGE_T[0]
	corner3[superpara.INPUT_SIZE - 1, 0] = 1

	corner4 = np.zeros((superpara.INPUT_SIZE, 1))
	corner4[0, 0] = superpara.RANGE_Y[1]
	corner4[1, 0] = superpara.RANGE_T[1]
	corner4[superpara.INPUT_SIZE - 1, 0] = 1

	datalist.append(corner1)
	datalist.append(corner2)
	datalist.append(corner3)
	datalist.append(corner4)
	"""

	return datalist




	
"""
#obsolete
#generating training set from uniform distribution
#uniform distribution
sample_t = np.random.uniform(superpara.RANGE_T[0] - superpara.EPS_T, superpara.RANGE_T[1] + superpara.EPS_T, size = (superpara.RAND_SIZE_T, ))
sample_t[0] = superpara.RANGE_T[0]
sample_t[-1] = superpara.RANGE_T[1]

sample_y = np.random.uniform(superpara.RANGE_Y[0]  - superpara.EPS_Y, superpara.RANGE_Y[1] + superpara.EPS_Y, size = (superpara.RAND_SIZE_Y, ))
sample_y[0] = superpara.RANGE_Y[0]
sample_y[-1] = superpara.RANGE_Y[1]
"""