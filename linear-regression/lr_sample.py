# Linear Regression Sample
# ========================================
# [] File Name : lr_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
import matplotlib.pyplot as plt 
import numpy as np 

dataset = np.array([[1,2.5],[2,3.5],[3,4.6],[4,4.8],[5,5.9],[6,7.1],[6.5,7.5]])

# Define the initial values
learning_rate = 0.02
numOfIterations = 2000
initialConstant = 5
initialSlope = 2

# ======================================== #
# ====== Train With Gradient Descent ===== #
# ======================================== #
def trainWithGradientDescent(coordiantes, h_slope, h_constant, learning_rate):

    # Indicates how much h values must be updated
    slope_gd_rate = 0
    constant_gd_rate = 0

    # Indicates the new slop and constant for each hypothesis
    updated_h_slope = h_slope
    updated_h_constant = h_constant

    # Repeat on each single data 
    # This is gradient descent obviously :)
    for i in range(0, len(coordiantes)):

        # Grab the current x and y
        x = coordiantes[i][0]
        y = coordiantes[i][1]

        constant_gd_rate += -2/len(coordiantes) * (y - (h_slope * x + h_constant))
        slope_gd_rate += -2/len(coordiantes) * x * (y - (h_slope * x + h_constant))
        print("Constant error: " + str(constant_gd_rate))
        print("Slope error: " + str(slope_gd_rate))

    updated_h_constant = h_constant - (learning_rate * constant_gd_rate)
    updated_h_slope = h_slope - (learning_rate * slope_gd_rate)
    

    return [updated_h_constant, updated_h_slope]

# ======================================== #
# ====== Initialize GD with values ======= #
# ======================================== #
def gradientDescentInitializer(cooridnates, initial_slope, initial_constant, learning_rate, numOfIterations):
    
    # First value for the trained constant
    # and trained slope
    trained_constant = initial_constant
    trained_slope = initial_slope
    # Start training for numOfIterations times
    for i in range(0, numOfIterations):
        trained_constant, trained_slope = trainWithGradientDescent(cooridnates, trained_slope, trained_constant, learning_rate)
    
    return [trained_constant, trained_slope]

# ======================================== #
# ======= Start & Plot the result ======== #
# ======================================== #
constant_result, slope_result = gradientDescentInitializer(dataset, initialSlope, initialConstant, learning_rate, numOfIterations)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset[:,0],dataset[:,1])
plt.plot(dataset[:,0], dataset[:,0] * slope_result + constant_result)
ax.text(0.95, 0.01, "Iterations: " + str(numOfIterations),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=10)
ax.text(0.95, 0.05, "Learning Rate: " + str(learning_rate),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=10)
plt.show()
