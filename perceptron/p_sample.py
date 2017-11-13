# Perceptron Sample
# ========================================
# [] File Name : p_sample.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
from numpy import array, random, dot
from random import choice

dataset = [         (array([3, 4, 5]), 1),
                    (array([3, 4, 6]), 0),
                    (array([3, 3, 5]), 0),
                    (array([3, 8, 5]), 0),
                    (array([4, 4, 5]), 1),
                    (array([1, 0, 5]), 1),
                    (array([7, 4, 8]), 1),
                    (array([1, 6, 4]), 0),
                    (array([3, 4, 5]), 0),
                    (array([4, 4, 6]), 1),
                    (array([9, 5, 5]), 0)
]

# Define the initial values
learning_rate = 0.2
weights = random.rand(3)
numOfIterations = 50

# Neuron decision function
activationFunction = lambda x: 0 if x < 0 else 1

# ======================================== #
# ========= Train The Perceptron ========= #
# ======================================== #
def trainSingleLayerPerceptron(dataset, weights, numOfIterations, learning_rate):
    for i in range(numOfIterations):
        # Select randomly from dataset
        inputVector, label = choice(dataset)

        # Find the dot product of weights4 and input vector
        result = dot(weights, inputVector)

        # Find the error
        resultError = label - activationFunction(result)

        # Update the weights
        weights += learning_rate * resultError * inputVector

    return weights

# Test the trained data (ignore the invalidation of this model for learning purpose)
learned_weights = trainSingleLayerPerceptron(dataset, weights, numOfIterations, learning_rate)

for vector_i, label_i in dataset:
    test_result = dot(learned_weights, vector_i)
    print("Classified ", vector_i, "as ", activationFunction(test_result))

