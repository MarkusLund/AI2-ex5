import math
import random
import copy

#The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0/(1.0+math.exp(-x)))

#The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return math.exp(-x)/(pow(math.exp(-x)+1,2))

def randomFloat(low,high):
    return random.random()*(high-low) + low

#Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0]*J)
    return m

class NN: #Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        #Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.inputActivation = [1.0]*self.numInputs
        self.hiddenActivations = [1.0]*self.numHidden
        self.outputActivation = 1.0 #Assuming a single output.
        self.learningRate = learningRate

        # create weights
        #A matrix with all weights from input layer to hidden layer
        self.weightsInput = makeMatrix(self.numInputs,self.numHidden)
        #A list with all weights from hidden layer to the single output neuron.
        self.weightsOutput = [0 for i in range(self.numHidden)]# Assuming single output
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weightsInput[i][j] = randomFloat(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weightsOutput[j] = randomFloat(-0.5, 0.5)

        #Data for the backpropagation step in RankNets.
        #For storing the previous activation levels (output levels) of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        if len(inputs) != self.numInputs-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations=copy.deepcopy(self.inputActivation)
        for i in range(self.numInputs-1):
            self.inputActivation[i] = inputs[i]
        self.inputActivation[-1] = 1 #Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations=copy.deepcopy(self.hiddenActivations)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                #print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.inputActivation[i] * self.weightsInput[i][j]
            self.hiddenActivations[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation=self.outputActivation
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.hiddenActivations[j] * self.weightsOutput[j]
        self.outputActivation = logFunc(sum)
        return self.outputActivation

    def computeOutputDelta(self):
        Pab = logFunc(self.prevOutputActivation-self.outputActivation)
        self.prevOutputActivation = logFuncDerivative(self.prevOutputActivation*(1-Pab))
        self.outputActivation = logFuncDerivative(self.outputActivation*(1-Pab))
        #TODO: Implement the delta function for the output layer (see exercise text)

    def computeHiddenDelta(self):
        #TODO: Implement the delta function for the hidden layer (see exercise text)
        for i in range(self.numHidden):

            out_ha = self.prevHiddenActivations[i]
            out_hb = self.hiddenActivations[i]
            w_ho = self.weightsOutput[i]
            diffOutputActiviation = self.prevOutputActivation-self.outputActivation

            self.prevDeltaHidden[i] = logFuncDerivative(out_ha)*w_ho*diffOutputActiviation
            self. deltaHidden[i] = logFuncDerivative(out_hb)*w_ho*diffOutputActiviation

    def updateWeights(self):
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weightsInput[i][j] = self.weightsInput[i][j] + self.learningRate * (self.prevHiddenActivations[j] * self.prevInputActivations[i] - self.hiddenActivations[j] * self.inputActivation[i])

        for j in range(self.numHidden):
            self.weightsOutput[j] = self.weightsOutput[j] + self.learningRate * (self.prevOutputActivation * self.prevHiddenActivations[j] - self.outputActivation * self.hiddenActivations[j])

        #TODO: Update the weights of the network using the deltas (see exercise text)

    def backpropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    #Prints the network weights
    def weights(self):
        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weightsInput[i])
        print()
        print('Output weights:')
        print(self.weightsOutput)

    def train(self, patterns, iterations=1):
        for i in patterns:
            self.propagate(i[0])
            self.propagate(i[1])
            self.backpropagate()
        #TODO: Train the network on all patterns for a number of iterations.
        #To measure performance each iteration: Run for 1 iteration, then count misordered pairs.
        #TODO: Training is done  like this (details in exercise text):
        #-Propagate A
        #-Propagate B
        #-Backpropagate

    def countMisorderedPairs(self, patterns):
        numRight = 0.0
        numMisses = 0.0
        for i in patterns:
            outputActivationA = self.propagate(i[0].features)
            outputActivationB = self.propagate(i[1].features)
            if outputActivationA > outputActivationB:
                winner = i[0]
                looser = i[1]
            else:
                winner = i[1]
                looser = i[0]

            if winner.rating > looser.rating:
                numRight += 1
            else:
                numMisses += 1
        print "NumMisses:", numMisses
        print "numRight:", numRight
        print numMisses / (numRight+numMisses)

        return numMisses / (numRight+numMisses)


        #TODO: Let the network classify all pairs of patterns. The highest output determines the winner.
        #for each pair, do
        #Propagate A
        #Propagate B
        #if A>B: A wins. If B>A: B wins
        #if rating(winner) > rating(loser): numRight++
        #else: numMisses++
        #end of for
        #TODO: Calculate the ratio of correct answers:
        #errorRate = numMisses/(numRight+numMisses)
