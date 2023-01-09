'''
Backpropagation implementation by Julian Robin (jr673)
to run, simply run the terminal command: python backProp.py
all data is enclosed within this file.
'''
import numpy as np
from random import seed
from random import random

#sigmoid function for use during activation calculation and its derivative for back propegation
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoidDerivative(x):
	return x * (1 - x)

epsilon = 0.1

#initialize arrays representing inputs and expected outputs as according to XOR truth table
inputArray=np.array([[1,1], [0,1], [1,0], [0,0]])
outputArray=np.array([[0], [1], [1], [0]])

inputDepth = 2
hiddenDepth = 2
outputDepth = 1

#initialize hidden and output layer weights with random values and print to console
#hiddenWeights = [[random() for i in range(inputDepth)] for i in range(hiddenDepth)]
hiddenWeights = np.random.uniform(size=(inputDepth, hiddenDepth))
#print("Hidden Layer Weights: ", hiddenWeights)

#NOTE: python list replaced with numpy array as numpy algebra, notably dot products, does
# not perform as expected on standard python lists. Transposing numpy arrays is also significantly easier.

#outputWeights = [[random() for i in range(hiddenDepth)] for i in range(outputDepth)]
outputWeights = np.random.uniform(size=(hiddenDepth,outputDepth))
#print("Output Layer Weights: ", outputWeights)


for _ in range(50000):

	#calculate activation of hidden and output neurons, performing a forward propegation
	hiddenActivation = np.dot(inputArray, hiddenWeights)
	hiddenOutput=sigmoid(hiddenActivation)

	outputActivation = np.dot(hiddenOutput, outputWeights)
	networkOutput = sigmoid(outputActivation)
	
	#Error Calculation using equations from lectures
	deltaK = (outputArray - networkOutput) * -1 * sigmoidDerivative(networkOutput)
	hiddenError = np.dot(deltaK, outputWeights.T) 
	deltaJ = hiddenError * sigmoidDerivative(hiddenOutput)

	#backpropegation using errors calculated
	outputWeights += np.dot(hiddenOutput.T, -deltaK)
	hiddenWeights += np.dot(inputArray.T, -deltaJ)
'''
print("hidden layer Weights: \n", hiddenWeights)

print("output layer weights: \n", outputWeights)

print("hidden layer activations: \n", hiddenOutput)

print("output layer activations: \n", networkOutput)
'''
print("Prediction for a = 1 & b = 1: ", int(np.round(networkOutput[0])))
print("Prediction for a = 0 & b = 1: ", int(np.round(networkOutput[1])))
print("Prediction for a = 1 & b = 0: ", int(np.round(networkOutput[2])))
print("Prediction for a = 0 & b = 0: ", int(np.round(networkOutput[3])))

