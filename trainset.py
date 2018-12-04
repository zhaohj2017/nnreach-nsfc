import superpara
import numpy as np

def gendata(step):
	datalist = []
	sample_y = []
	sample_t = []

	#very important!!!
	#update T-RANGE
	t_start = step * superpara.T_STEP - superpara.EPS_T
	t_end = (step + 1) * superpara.T_STEP + superpara.EPS_T
	#sample point by meshing
	sample_t = np.arange(t_start, t_end + superpara.MESH_SIZE_T * 0.99, superpara.MESH_SIZE_T) 
		#plus superpara.MESH_SIZE_T * 0.99 to include the right end point

	enlarge_y = step * superpara.ENLARGE_Y
	y_start = superpara.RANGE_Y[0] - enlarge_y - superpara.EPS_Y
	y_end = superpara.RANGE_Y[1] + enlarge_y + superpara.EPS_Y
	sample_y = np.arange(y_start, y_end + superpara.MESH_SIZE_Y * 0.99, superpara.MESH_SIZE_Y)

	if len(sample_y) == 0: # a single point inital y
		sample_y = np.array([superpara.RANGE_Y[0]])

	for st in sample_t:
		for sy in sample_y:
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[0, 0] = sy
			data[1, 0] = st
			data[superpara.INPUT_SIZE - 1, 0] = 1
			datalist.append(data)
		#sample_y = np.fliplr([sample_y])[0] #reverse: not helpful? discard!

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