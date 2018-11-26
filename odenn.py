import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test
import chkweight



#generate train set data
traindata = trainset.gendata()

#train the network
backward.itrdescent(traindata)


#check weight
chkweight.outweight()

#check the precision
test.chkprecision()
#test.reachplot(superpara.MESH_SIZE_Y / 10.0, superpara.MESH_SIZE_T / 10.0)
