# Artificial Neural Network
# ========================================
# [] File Name : ann_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
import math
from numpy import array, random, dot


# The dataset from the perceptron (which it was failing at classification procedure)
dataset = array([   
                    [3, 4, 5],
                    [3, 4, 6],
                    [3, 3, 5],
                    [3, 8, 5],
                    [4, 4, 5],
                    [1, 0, 5],
                    [7, 4, 8],
                    [1, 6, 4],
                    [3, 4, 5],
                    [4, 4, 6],
                    [9, 5, 5]
])

dataset_labels = array([
    1, 0, 0 , 0 , 1, 1, 1, 0, 0, 1, 0
])

# Define the initial values
numOfIterations = 5000

# Initial weights
firstLayerWeights = random.rand(3, 2)
secondLayerWeights = random.rand(2, 1)

# The beloved sigmoid function 
def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

# The sigmoid curve
def sigmoidCurve(x):
    return x * (1 - x)

# ======================================== #
# ======== Train The 2 Layer ANN ========= #
# ======================================== #
def trainNeuralNetwork(dataset, flw, slw, numOfIterations):
    
    
    updatedFLW = 0
    updatedSLW = 0

    for i in range(numOfIterations):
        firstLayerOutputVector = sigmoid(dot(flw, dataset))
        secondLayerOutputVector = sigmoid(dot(slw, firstLayerOutputVector))

        # Find the error
        secondLayerError = secondLayerOutputVector - dataset[:, 1]

        # Calculate the adjustment value for the layer 2 weights
        secondLayerWeightAdjustmentValue = secondLayerError * secondLayerOutputVector * sigmoidCurve(secondLayerOutputVector)

        # Update the layer 2 weights
        slw += secondLayerWeightAdjustmentValue

        # First layer error
        firstLayerError = secondLayerWeightAdjustmentValue.dot(slw.T)

        # First layer weight adjustment
        firstLayerWeightAdjustmentValue = firstLayerError * firstLayerOutputVector * sigmoidCurve(firstLayerOutputVector)

        # Update the layer 1 weights
        flw += firstLayerWeightAdjustmentValue

        updatedFLW = flw
        updatedSLW = slw
    
    return [flw, slw]


# ======================================== #
# ======= Start and Plot The Result ====== #
# ======================================== #

print(sigmoid(dot(dataset, firstLayerWeights)))

# trainNeuralNetwork(dataset, firstLayerWeights, secondLayerWeights, numOfIterations)