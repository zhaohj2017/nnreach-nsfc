import numpy as np
import superpara
import gradient
import ann
import backward
import trainset
import test


"""
To solve a problem:
1. Modify ode.py
2. Modify superpara.py
3. Modify adaptive.py
4. Modify in test.py the check precision method, i.e. chkprecision()
5. Determine in odenn.py whether you want to check precision or output the reachable set plotting: chkprecision() or reachplot()
"""



#generate train set data
traindata = trainset.gendata()

#train the network
backward.itrdescent(traindata)

#check the precision
test.chkprecision()
test.reachplot(superpara.MESH_SIZE_Y / 10.0, superpara.MESH_SIZE_T / 10.0)

print ann.weight_h_o
print ann.weight_y_h
print ann.weight_t_h
print ann.weight_b_h
