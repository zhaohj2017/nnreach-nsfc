import superpara

#*********************   dy / dt = exp(y) ****************************	
#adjust learn rate or restart for example: dy/dt = exp(y)
def jump(curr, delta):
	if curr > 1e0 and delta < 1e-2:
		print "RESTARAT!!!\n"
		#return from gradescent early, 0 means that the loop in itrdescent continues, but stop a new loop (restart)
		return 0
	else:
		return 1 #do not restart

#rate adjustment for example: dy/dt = exp(y)
def adjust(curr, delta):
	if curr < 1e-1 and delta < 1e-2: #continue to the next epoch
		superpara.LEARN_RATE = - 1
		superpara.BATCH_SIZE = 1

	
"""
#*********************   dy / dt = y ****************************

#adjust learn rate or restart for example: dy/dt = y
def jump(curr, delta):
	if curr > 1e0 and delta < 1e-2:
		print "RESTARAT!!!\n"
		return 0 #return from gradescent early, 0 means that the loop in itrdescent continues, but stop a new loop (restart)

#rate adjustment for example: dy/dt = y
def adjust(curr, delta):
	if curr < 1e-1 and delta < 1e-2: #continue to the next epoch
		superpara.LEARN_RATE = - 1
		superpara.BATCH_SIZE = 1
"""