import numpy as np
import superpara
import gradient
import ann
import backward

#train
ann.input[0, 0] = superpara.RANGE_Y[0]
ann.input[1, 0] = superpara.RANGE_T[0]
ann.input[2, 0] = 1
backward.gdescent(ann.input)


#test
ann.input[1, 0] = superpara.RANGE_T[1] - superpara.EPSILON
ann.input[0, 0] = superpara.RANGE_Y[0] + superpara.EPSILON

res = []
while ann.input[0, 0] <= superpara.RANGE_Y[1] - superpara.EPSILON:
	res.append(gradient.sol_candidate(ann.input))
	ann.input[0, 0] += 0.01

#output
print "test:", max(res), "\treal:", (superpara.RANGE_Y[1] - superpara.EPSILON) * np.exp(ann.input[1, 0])
print "test:", min(res), "\treal:", (superpara.RANGE_Y[0] + superpara.EPSILON) * np.exp(ann.input[1, 0])
