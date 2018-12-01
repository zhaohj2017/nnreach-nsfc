#working 
#super parameter for example: dy / dt 

EPOCHS = 100
LEARN_RATE = - 0.01

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2


NUM_HIDDEN = 5

BATCH_SIZE = 1
BATCH_NUM = 0


MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPS_Y = 0.05
EPS_T = 0.05 # EPS_T should be less than T_STEP

RANGE_Y = [0, 1]

LENGTH_T = 0.3
RANGE_T = [0, LENGTH_T]

T_STEP = 0.3
NUM_STEP = int(round((RANGE_T[1] - RANGE_T[0]) / T_STEP)) #very important
T_START = 0

TEST_RATE = 10

PLOT_MESH_Y = MESH_SIZE_Y / TEST_RATE
PLOT_MESH_T = MESH_SIZE_T / TEST_RATE







""" **********  NOT RESOLVED!!!  **********************
#**************    dy / dt = exp(y)  *********************************

#super parameter for example: dy / dt = exp(y)
#comparison with Flow*

EPOCHS = 10
LEARN_RATE = -1

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2

MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPS_Y = 0.05
EPS_T = 0.05 # EPS_T should be less than T_STEP

RANGE_Y = [0, 1]

LENGTH_T = 1
RANGE_T = [0, LENGTH_T]

NUM_HIDDEN = 10

BATCH_SIZE = 20
BATCH_NUM = 0

T_STEP = 1
NUM_STEP = int((RANGE_T[1] - RANGE_T[0]) / T_STEP)
T_START = 0
"""


""" ********************* FIXED !!! DO NOT CHANGE !!! ************************
#**************  dy  /dt = y (using sigmoid) *************************

#super parameter for example: dy / dt = y
EPOCHS = 10
LEARN_RATE = - 1

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2

MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPS_Y = 0.05
EPS_T = 0.05 # EPS_T should be less than T_STEP

RANGE_Y = [0, 1]

LENGTH_T = 1
RANGE_T = [0, LENGTH_T]

NUM_HIDDEN = 5

BATCH_SIZE = 1
BATCH_NUM = 0

T_STEP = 1
NUM_STEP = int((RANGE_T[1] - RANGE_T[0]) / T_STEP)
T_START = 0
"""



"""******* FIXED for fast sigmoid with dy / dt = y **************
EPOCHS = 3

LEARN_RATE = - 0.18

DIMENSON = 1
INPUT_SIZE = DIMENSON + 2

MESH_SIZE_Y = 0.01
MESH_SIZE_T = 0.01

EPS_Y = 0.05
EPS_T = 0.05 # EPS_T should be less than T_STEP

RANGE_Y = [0, 1]

LENGTH_T = 1
RANGE_T = [0, LENGTH_T]

NUM_HIDDEN = 130

BATCH_SIZE = 1
BATCH_NUM = 0

T_STEP = 1
NUM_STEP = int((RANGE_T[1] - RANGE_T[0]) / T_STEP)
T_START = 0
"""
