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
import time

time_start = time.time()

#generate train set data
for step in range(superpara.NUM_STEP):
	#generating training set
	dataset = trainset.gendata(step) #generating training set for every time step
	#when start a new time step, do we need to update the weight matrix?
	backward.itrdescent(dataset, step)
	pipes.addpipe()
	test.chkprecision(step)
	#chkweight.outweight()

time_end = time.time()

#plot
plot.horiplot(superpara.PLOT_MESH_Y, superpara.PLOT_MESH_T)

##pause
#raw_input() ##for python 3 use 'input()'
#plot.vertiplot(superpara.PLOT_MESH_Y, superpara.PLOT_MESH_T)

print "\ntotally cost (excluding plotting):", time_end - time_start