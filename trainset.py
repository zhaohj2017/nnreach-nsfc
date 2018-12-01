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
	sample_t = np.arange(superpara.RANGE_T[0] - superpara.EPS_T, superpara.RANGE_T[1] + superpara.EPS_T, superpara.MESH_SIZE_Y)
	
	sample_y = np.arange(superpara.RANGE_Y[0] - superpara.EPS_Y, superpara.RANGE_Y[1] + superpara.EPS_Y, superpara.MESH_SIZE_Y)
	if len(sample_y) == 0:
		sample_y = np.array([superpara.RANGE_Y[0]])

	for sy in sample_y:
		for st in sample_t:
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[0, 0] = sy
			data[1, 0] = st
			data[superpara.INPUT_SIZE - 1, 0] = 1
			datalist.append(data)
		sample_t = np.fliplr([sample_t])[0] #reverse
		
	#print datalist

	return datalist

	
