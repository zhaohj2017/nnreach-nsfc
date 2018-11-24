import superpara
import numpy as np

def gendata():
	datalist = []
	sample_y = []
	sample_t = []

	n_sample_y = int((superpara.RANGE_Y[1] - superpara.RANGE_Y[0]) / superpara.MESH_SIZE_Y) + 1
	n_sample_t = int((superpara.RANGE_T[1] - superpara.RANGE_T[0]) / superpara.MESH_SIZE_T) + 1

	for ny in range(n_sample_y):
		sample_y.append(superpara.RANGE_Y[0] + superpara.MESH_SIZE_Y * ny)
	for nt in range(n_sample_t):
		sample_t.append(superpara.RANGE_T[0] + superpara.MESH_SIZE_T * nt)

	for ny in range(n_sample_y):
		for nt in range(n_sample_t):
			data = np.zeros((superpara.INPUT_SIZE, 1))
			data[superpara.INPUT_SIZE - 1, 0] = 1
			data[0, 0] = sample_y[ny]
			data[1, 0] = sample_t[nt]
			datalist.append(data)

	return datalist

