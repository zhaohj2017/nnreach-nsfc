import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test
import random

#generate train set data
traindata = trainset.gendata()

#train the network
random.shuffle(traindata)
print traindata
backward.gdescent(traindata)

#check the precision
test.gentest()
