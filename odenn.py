import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test


#generate train set data
traindata = trainset.gendata()

#train the network
backward.recdescent(traindata)

#check the precision
test.gentest()
