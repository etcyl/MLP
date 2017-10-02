# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 20:42:26 2016

@author: Etcyl
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

class neuron(object):
    """A single neuron is a computational element"""
    #The neuron computes the sum of the input
    
    type = 'Neuron'
    
    def __init__(self, numInputs):
        self.numInputs = numInputs
        self.inputs = [0]*self.numInputs
        self.output = 0
        
    #Non-linear computation methods
    def update(self):
        self.output = np.sum(self.inputs)
        
    #printState 
    def printState(self):
        return self.inputs, self.output 
        
    #inputs methods
    def setInput(self, inputNum, value):
        self.inputs.insert(inputNum, value)
    
    def getInput(self):
        return self.copy
        
    def getOutput(self):
        return self.output
        
    def reset(self):
        self.inputs = []
        
class sNeuron(neuron):
    """A sigmoidal neuron is a non-linear mapping machine"""
    
    type = 'Sigmoidal Neuron'
    
    def update(self):
        self.sum = np.sum(self.inputs)
        self.output = sigmoid(self.sum)
        
class connectionLink(object):
    """A connection class stores a weight and a source and dest ID"""
    #The class performs the dot product on user set weights 
    #with the input
    
    type = 'Connection class'
    
    def __init__(self, weight, sourceID, destID):
        self.weight = weight
        self.sourceID = sourceID
        self.destID = destID
    
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight 
     
    def getSourceID(self):
        return self.sourceID
   
    def setSourceID(self, sourceID):
       self.sourceID = sourceID
       
    def getDestID(self):
        return self.destID
    
    def setDestID(self, destID):
        self.destID = destID
        
    def printState(self):
        print (self.weight, self.sourceID, self.destID)

class network(object):
    """A network is a core of neurons with connections"""
    #Network makes connections and neurons using an adjacency matrix
    
    type = 'Network'
    
    def __init__(self):
        self.numNeurons = 0
        self.neurons = []
        self.inputs = []
        self.links = []
        self.sNeurons = []
        
    def createNetwork(self, adjMtx):
        self.networkMtx = adjMtx
        self.numNeurons = len(adjMtx)
        self.hiddenLinks = []
        self.outputLinks = []
        for j in range(self.numNeurons):
            inputCounter = 0
            for i in range(self.numNeurons):
                if adjMtx[i][j] != 0:
                    c = connectionLink(adjMtx[i][j], i, j)
                    c.attr = i
                    self.links.append(c)
                    inputCounter = inputCounter + 1
            if inputCounter == 0:
                inputCounter = 1
                n = neuron(inputCounter)
                self.inputs.append(n)
            else:
                n = sNeuron(inputCounter)
                n.attr = j
                self.sNeurons.append(n)
            self.neurons.append(n)
        for link in self.links:
            if link.getDestID() == 5:
                self.outputLinks.append(link)
            else:
                self.hiddenLinks.append(link)
                          
    def printState(self):
        print('Number of neurons: ')
        print(self.numNeurons)
        print
        print('Neuron list: ')
        print(self.neurons)
        print('Link list: ')
        print(self.links)
    
    def forwardProp(self):
        count = 0
        temp = 0
        for link in self.links:
            if count == 0:
                temp = 0
            elif count == 1:
                temp = 1
            elif count == 2:
                temp = 0
            elif count == 3:
                temp = 1
            elif count == 4:
                temp = 0
            elif count == 5:
                temp = 1
            elif count == 6:
                temp = 0
            elif count == 7:
                temp = 1
            elif count == 8:
                temp = 2
            source = link.getSourceID()
            out = self.neurons[source].getOutput()
            weight = link.getWeight()
            out = out*weight
            destination = link.getDestID()
            self.neurons[destination].inputs[temp] = out
            count = count + 1
            n.neurons[destination].update()
        count = 0
  
class backpropagate(object):
    """A backpropagation algorithm modifies weights of a network"""
    #Computes the gradient of the cost function
    #Determines error
    #Corrects weight based off of the error, with respect to (wrt) every weight
    
    type = 'Backpropagation training algorithm'
    
    def __init__(self):
        self.error = 0
    
    def train(self, inputs, targets, network):
        eta = 0.5   #Training rate
        self.newLinks = [0]*9
        self.x = random.randint(0, 3)    #Get a random row number to use for the input / target patterns
        network.neurons[0].inputs[0] = inputs[self.x][0]   #Set the first two inputs
        network.neurons[1].inputs[0] = inputs[self.x][1]
        network.neurons[0].update()     #And update them so they outputs
        network.neurons[1].update()
        network.forwardProp()       #Propagate the input pattern through the network
        out = network.neurons[-1].getOutput()       #Get the network output
        self.error = abs(out - targets[self.x][0])
        pdout = -(targets[self.x][0] - out)      #How much tot error changes wrt output
        delta = pdout*(out*(1 - out))       #Delta rule
        #Hidden layer weight corrections
        count = 0
        for i in range(6):
            if i == 0 or i == 1:
                count = 6
            elif i == 2 or i == 3:
                count = 7
            elif i == 4 or i == 5:
                count = 8                
            weight = network.links[i].getWeight()
            oWeight = network.links[count].getWeight()
            source = network.links[count].getSourceID()
            hOut = network.neurons[source].getOutput()
            iSource = network.links[i].getSourceID()
            iOut = network.neurons[iSource].getOutput()
            newWeight = weight - eta*delta*oWeight*(hOut*(1 - hOut))*iOut
            self.newLinks[i] = newWeight
        #Output layer weight corrections
        hOut = network.neurons[2].getOutput()
        self.newLinks[6] = network.links[6].getWeight() - delta*eta*hOut
        hOut = network.neurons[3].getOutput()
        self.newLinks[7] = network.links[7].getWeight() - delta*eta*hOut
        hOut = network.neurons[2].getOutput()
        self.newLinks[8] = network.links[8].getWeight() - delta*eta*hOut
        for i in range(9):
            network.links[i].setWeight(self.newLinks[i])
            
#END OF CLASS DEFINITIONS
            
def sigmoid(x):
    return 1/(1 + math.exp(-x))
            
def forwardProp(n, x):
    n.neurons[0].inputs[0] = inputs[x][0]   #Set the first two inputs
    n.neurons[1].inputs[0] = inputs[x][1]
    n.neurons[0].update()     #And update so the input neurons have outputs
    n.neurons[1].update()
    count = 0
    temp = 0
    for link in n.links:
        if count == 0:
            temp = 0
        elif count == 1:
            temp = 1
        elif count == 2:
            temp = 0
        elif count == 3:
            temp = 1
        elif count == 4:
            temp = 0
        elif count == 5:
            temp = 1
        elif count == 6:
            temp = 0
        elif count == 7:
            temp = 1
        elif count == 8:
            temp = 2
        source = link.getSourceID()
        out = n.neurons[source].getOutput()
        weight = link.getWeight()
        out = out*weight
        destination = link.getDestID()
        n.neurons[destination].inputs[temp] = out
        count = count + 1
        n.neurons[destination].update()
    count = 0    
        
def printState(n):
    for x in range(len(n.links)):
        print
        print 'The state (weight, sourceID, destID) of link ', x, 'is', n.links[x].printState()
        print
    for x in range(len(n.neurons)):
        print
        print 'The state (input(s), output) of neuron ', x, 'is', n.neurons[x].printState()
        print
        
def trainAndPrint(inputs, targets, n, epoch):
    for x in range(epoch):
        b.train(inputs, targets, n)
        printState(n)
        
def trainAndPlot(inputs, targets, n, b):
    i = [0]*10
    error = []
    y = []
    iterations = []
    for x in range(10):
        trainAndPrint(inputs, targets, n, 1)
        i[x] = 1
        iterations.append(x)
        error.append(b.error)
        y.append(b.error)
    count = 10
    #while sum(error)/(sum(i)) >= .01:
    while count <= 20000:
        for x in range(10):
            trainAndPrint(inputs, targets, n, 1)
            i[x] = 1
            error[x] = b.error
            iterations.append(count)
            y.append(b.error)
            count = count + 1
    plt.xlabel('Number of iterations')
    plt.ylabel('Average error')
    plt.title('~ A training: average error versus training iterations')
    plt.plot(iterations, y)
    
adjMtx = np.random.random((6, 6))*0
adjMtx[0, 2] = random.random()
adjMtx[0, 3] = random.random()
adjMtx[0, 4] = random.random()
adjMtx[1, 2] = random.random()
adjMtx[1, 3] = random.random()
adjMtx[1, 4] = random.random()
adjMtx[2, 5] = random.random()
adjMtx[3, 5] = random.random()
adjMtx[4, 5] = random.random()

#Binary inputs
inputs = np.random.random((4, 2))*0
inputs[1][1] = 1
inputs[3][0] = 1
inputs[2][0] = 1
inputs[3][1] = 1

#Targets
targets = np.random.random((4, 1))*0
targets[0][0] = 1
targets[1][0] = 1

n = network()

n.createNetwork(adjMtx)

b = backpropagate()