# Gradient Descent Sample
# ========================================
# [] File Name : gd_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
import numpy as np
import matplotlib.pyplot as plt

# Initial Definitions
current_x = 0.5     
learning_rate = 0.02
num_iterations = 150

# Goal Function is 5x^4 - 6x^2
def findSlopeAtGivenPoint(x):
    return 5 * x ** 4 - 6 * x ** 2

# ======================================== #
# ==  Train The Gradient Descent & Plot == #
# ======================================== #
def trainGradientDescent(iter_range, current_x):

    # Gradient Descent Result Array
    gd_result = []
    for i in range(iter_range):
        previous_x = current_x
        current_x += -learning_rate * findSlopeAtGivenPoint(current_x)
        gd_result.append(current_x)
        print("X was updated to: ", previous_x)

    return gd_result

# ======================================== #
# ====  Points to Plot The GD Diagram ==== #
# ======================================== #
gradientDescentResult = trainGradientDescent(num_iterations, current_x)

fig = plt.figure()
fig.suptitle('Gradient Descent of Y = X^5 - 2X^3 - 2', fontsize=10, fontweight='bold')
ax = fig.add_subplot(1,1,1)
ax.title.set_text("Gradient Descent Result")

ax.plot(gradientDescentResult)
plt.show()