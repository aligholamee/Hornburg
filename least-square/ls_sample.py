# Least Square Sample
# ========================================
# [] File Name : ls_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#

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
print("Hypothesis y = 5x + 6 error: ")
print(computeErrorForLineGivenPoints(5, 6, [[3,5],[5,3],[8,4],[3,1],[6,4]]))

print("Hypothesis y = 3x + 1 error: ")
print(computeErrorForLineGivenPoints(3, 1, [[3,5],[5,3],[8,4],[3,1],[6,4]]))

print("Hypothesis y = 6x + 4 error: ")
print(computeErrorForLineGivenPoints(6, 4, [[3,5],[5,3],[8,4],[3,1],[6,4]]))

print("Hypothesis y = 6x + 8 error: ")
print(computeErrorForLineGivenPoints(6, 8, [[3,5],[5,3],[8,4],[3,1],[6,4]]))

print("Hypothesis y = 3x + 4 error: ")
print(computeErrorForLineGivenPoints(3, 4, [[3,5],[5,3],[8,4],[3,1],[6,4]]))