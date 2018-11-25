#**************    dy / dt = exp(y)  *********************************

#super parameter for example: dy / dt = exp(y)
#comparison with Flow*

EPOCHS = 2000
LEARN_RATE = - 1

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2

MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPSILON = 0.05

RANGE_Y = [0 - EPSILON, 1 + EPSILON]
RANGE_T = [0 - EPSILON, 0.3 + EPSILON]

NUM_HIDDEN = 5

BATCH_SIZE = 100
BATCH_NUM = 0





"""
#**************  dy  /dt = y  ******************************************

#super parameter for example: dy / dt = y
EPOCHS = 10
LEARN_RATE = - 1

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2

MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPSILON = 0.05

RANGE_Y = [0 - EPSILON, 1 + EPSILON]
RANGE_T = [0 - EPSILON, 1 + EPSILON]

NUM_HIDDEN = 5

BATCH_SIZE = 1
BATCH_NUM = 0
"""