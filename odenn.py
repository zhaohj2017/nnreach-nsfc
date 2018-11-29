import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test
import chkweight
import pipes



#generate train set data


for step in range(superpara.NUM_STEP):
	traindata = trainset.gendata(step)

	#when start a new time step, do we need to update the weight matrix?
	backward.itrdescent(traindata, step)

	pipes.addpipe()

	test.chkprecision(step)

#plot
test.reachplot(superpara.MESH_SIZE_Y / 5.0, superpara.MESH_SIZE_T / 5.0)
