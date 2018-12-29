import superpara

#************************* working **********************************************
#adjust learn rate or restart for example: dy / dt = exp(y)
def jump(curr, delta):
	# if curr > 1e0 and delta < 1e-2:
	# 	print "RESTARAT!!!\n"
	# 	#return from gradescent early, 1 means that the loop in itrdescent continues, but stop a new loop (restart)
	# 	return 1
	# else:
	# 	return 0 #do not restart
	pass

#rate adjustment for example: dy / dt = exp(y)
def adjust(curr, delta):
	"""
	if curr < 1e-1 and delta < 1e-2: #continue to the next epoch
		superpara.LEARN_RATE = 1
		superpara.BATCH_SIZE = 1
	"""
	pass

def stop(curr, delta):
	if curr < 1e-4: #error < 0.0001
		print "Success! Stop!"
		return 1


"""
#*********************   dy / dt = exp(y) ****************************
#adjust learn rate or restart for example: dy/dt = exp(y)
def jump(curr, delta):
	if curr > 1e0 and delta < 1e-2:
		print "RESTARAT!!!\n"
		#return from gradescent early, 1 means that the loop in itrdescent continues, but stop a new loop (restart)
		return 1
	else:
		return 0 #do not restart

#rate adjustment for example: dy/dt = exp(y)
def adjust(curr, delta):
	if curr < 1e-1 and delta < 1e-2: #continue to the next epoch
		superpara.LEARN_RATE = 1
		superpara.BATCH_SIZE = 1	
"""



"""
#************* FIXED DO NOT CHANGE !!! ***********************
#*********************   dy / dt = y ****************************

#adjust learn rate or restart for example: dy / dt = y
def jump(curr, delta):
	if curr > 1e0 and delta < 1e-2:
		print "RESTARAT!!!\n"
		return 1 #return from gradescent early, 1 means that the loop in itrdescent continues, but stop a new loop (restart)

#rate adjustment for example: dy / dt = y
def adjust(curr, delta):
	if curr < 1e-1 and delta < 1e-2: #continue to the next epoch
		superpara.LEARN_RATE = 1
		superpara.BATCH_SIZE = 1
"""



"""
## using fast sigmoid
#*********************   dy / dt = y ****************************

#adjust learn rate or restart for example: dy / dt = y
def jump(curr, delta):
	if curr > 1e0 and delta < 1e-2:
		print "RESTARAT!!!\n"
		return 1 #return from gradescent early, 1 means that the loop in itrdescent continues, but stop a new loop (restart)

#rate adjustment for example: dy / dt = y
def adjust(curr, delta):
	if curr < 0.005: #continue to the next epoch
		superpara.LEARN_RATE = 0.18
		superpara.BATCH_SIZE = 1
"""