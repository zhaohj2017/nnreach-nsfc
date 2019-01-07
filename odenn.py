import numpy as np
import superpara
import gradient
import ann
import trainset
import test
import pipes
import plot
import time
import bfgs

time_start = time.time()

#generate train set data
for step in range(superpara.NUM_STEP):
	#generating training set
	dataset = trainset.gendata(step) #generating training set for every time step

	#superpara.LEARN_RATE = 0.5
	superpara.EPOCHS = 1
	superpara.BATCH_SIZE = len(dataset)
	superpara.BFGS_BATCH_ITR_NUM = 1000
	superpara.PRINT_MINI = 1
	bfgs.itrdescent(dataset, step)

	#learned a pipe segment
	pipes.addpipe()
	#check the precision for this step
	test.chkprecision(step)
	#ann.outweight()

time_end = time.time()

#plot
#plot.horiplot()
plot.vertiplot()

##pause
#raw_input() ##for python 3 use 'input()'

print "\ntotally cost (excluding plotting):", time_end - time_start