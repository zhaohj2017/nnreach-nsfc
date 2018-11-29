import superpara
import numpy as np

def gendata(step):
	datalist = []
	sample_y = []
	sample_t = []

	#very important!!!
	#update T-RANGE
	superpara.RANGE_T[0] = step * superpara.T_STEP
	superpara.RANGE_T[1] = (step + 1) * superpara.T_STEP

	n_sample_y = int((superpara.RANGE_Y[1] - superpara.RANGE_Y[0] + 2 * superpara.EPS_Y) / superpara.MESH_SIZE_Y) + 1
	n_sample_t = int((superpara.RANGE_T[1] - superpara.RANGE_T[0] + 2 * superpara.EPS_T) / superpara.MESH_SIZE_T) + 1

	for ny in range(n_sample_y):
		sample_y.append(superpara.RANGE_Y[0] - superpara.EPS_Y + superpara.MESH_SIZE_Y * ny)
	for nt in range(n_sample_t):
		sample_t.append(superpara.RANGE_T[0] - superpara.EPS_T + superpara.MESH_SIZE_T * nt)



	sample_y = np.arange(superpara.RANGE_Y[0] - superpara.EPS_Y, superpara.RANGE_Y[1] + superpara.EPS_Y, superpara.MESH_SIZE_Y)
	sample_t = np.arange(superpara.RANGE_T[0] - superpara.EPS_T, superpara.RANGE_T[1] + superpara.EPS_T, superpara.MESH_SIZE_Y)




	"""
	for ny in range(n_sample_y):
		for nt in range(n_sample_t):
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[superpara.INPUT_SIZE - 1, 0] = 1
			data[0, 0] = sample_y[ny]
			data[1, 0] = sample_t[nt]
			datalist.append(data)
	"""

	for sy in sample_y:
		for st in sample_t:
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[0, 0] = sy
			data[1, 0] = st
			data[superpara.INPUT_SIZE - 1, 0] = 1
			datalist.append(data)
		sample_t = np.fliplr([sample_t])[0]
		
	#print datalist
	return datalist

	
