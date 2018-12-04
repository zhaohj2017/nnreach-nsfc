# odenn
A neural network application for engineering.


To solve a problem:
1. Modify ode.py
2. Modify superpara.py
3. Modify activation.py: choose an activation function
4. Modify adaptive.py
5. Modify in test.py the check precision method, i.e. chkprecision(), the real values
6. Modify in plot.py the plot of real upper and lower bounds


#revise plot-vertically !!!

#update delta matrix: not helpful
#quasi-newton: to be implemented
#simulation adaptive step + sensitivey analysis
#increment the range of y at the beginning of each step: to be. not a very important issue
#for every epoch, how the weight change between two random point? No relation
#for adjacent points, will the weight be close to each other? ODE of weights? no!
#compare with taylor expansion? done!
#initial weight distribution
#regularization term
#fast computation methods (matrix) for mini batches
#ode45 sampling points, higher order
#dropout: change the network topology