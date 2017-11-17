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
from numpy import array, random, dot, exp


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

dataset_labels = array([[1], [0], [0], [0], [1], [1], [1], [0], [0], [1], [0]])

# Define the initial values
numOfIterations = 5000

# Initial weights
firstLayerWeights = random.rand(3, 2)
secondLayerWeights = random.rand(2, 1)

# The beloved sigmoid function
def sigmoid(x):
    spart = (1 / (1 + exp(-x)))
    return spart

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
        firstLayerOutputVector = sigmoid(dot(dataset, flw))
        secondLayerOutputVector = sigmoid(dot(firstLayerOutputVector, slw))

        # Find the error and delta for the final layer
        secondLayerError = secondLayerOutputVector - dataset_labels
        secondLayerDelta = secondLayerError * sigmoidCurve(secondLayerOutputVector)

        # Find the error and delta for the first layer
        firstLayerError = secondLayerDelta.dot(slw.T)
        firstLayerDelta = firstLayerError * sigmoidCurve(firstLayerOutputVector)

        # Update the layer 1 and layer 2 weights
        flw -= secondLayerOutputVector.T.dot(firstLayerDelta)
        slw -= firstLayerOutputVector.T.dot(secondLayerDelta)

        updatedFLW = flw
        updatedSLW = slw
    
    return [flw, slw]


# ======================================== #
# ======= Start and Plot The Result ====== #
# ======================================== #
l1_weight, l2_weight = trainNeuralNetwork(dataset, firstLayerWeights, secondLayerWeights, numOfIterations)
print("Layer 1 weight updated to:")
print(l1_weight)

print("\nLayer 2 weight updated to:")
print(l2_weight)