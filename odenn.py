import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test
import chkweight
import pipes
import plot


#generate train set data
for step in range(superpara.NUM_STEP):
	#generating training set
	dataset = trainset.gendata(step)
	#when start a new time step, do we need to update the weight matrix?
	backward.itrdescent(dataset, step)
	pipes.addpipe()
	test.chkprecision(step)

#plot
plot.reachplot(superpara.PLOT_MESH_Y, superpara.PLOT_MESH_T)