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
import bfgs

time_start = time.time()

#generate train set data
for step in range(superpara.NUM_STEP):
	#generating training set
	dataset = trainset.gendata(step) #generating training set for every time step
	#stochastic gradient descent
	backward.itrdescent(dataset, step) #first sgd, and then bfgs

	#bfgs: only works in batch mode
	superpara.EPOCHS = 500
	superpara.BATCH_SIZE = len(dataset)
	superpara.LEARN_RATE = 0.05
	bfgs.itrdescent(dataset, step)
	superpara.LEARN_RATE = 0.1
	superpara.EPOCHS = 500

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