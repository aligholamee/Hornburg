# Least Square Sample
# ========================================
# [] File Name : ls_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
import matplotlib.pyplot as plt
import numpy as numpy

dataset = numpy.array([[3,5],[5,3],[8,4],[3,1],[6,4],[5,4],[7,5],[8,3]])

slope_list = [5, 3, 6, 6, 3, 4]
constant_list = [6, 1, 4, 8, 4, 7]

plot_titles = [
    'y = 5x + 6',
    'y = 3x + 1',
    'y = 6x + 4',
    'y = 6x + 8',
    'y = 3x + 4',
    'y = 4x + 7'
]

# ======================================== #
# ========== Least Square Error ========== #
# ======================================== #
def computeErrorForLineGivenPoints(b, m, coordinates):
    totalError = 0
    
    for i in range(0, len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]

        # Calcuate the error
        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(coordinates))

# ======================================== #
# ============ Test with data ============ #
# ======================================== #
errorlist = []

for i in range(0, 6):
    errorlist.append(computeErrorForLineGivenPoints(slope_list[i], constant_list[i], dataset))
    print("Hypothesis " + plot_titles[i] + " error: ")
    print(errorlist[i])

# ======================================== #
# ============ Plot the result =========== #
# ======================================== #
fig = plt.figure()
fig.suptitle('Least Square Errors', fontsize=10, fontweight='bold')

for i in range(1, 7):
    ax = fig.add_subplot(3, 2, i)
    ax.title.set_text(plot_titles[i-1])
    ax.scatter(dataset[:,0],dataset[:,1])  

    errorLabel = "Error = "
    ax.text(0.95, 0.01, errorLabel + str(errorlist[i-1]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=12)
    plt.plot(dataset, dataset/slope_list[i-1] + constant_list[i-1])

plt.show()