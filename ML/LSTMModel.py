import numpy as np
import math

class LSTMModel:
    def __init__(self, inLayerNumber, middleLayersNumber):
        self.inLayerNumber = inLayerNumber
        self.middleLayersNumber = middleLayersNumber
        self.middleLayerSts = np.zeros((self.middleLayersNumber))
        self.stateChanges = np.zeros((self.middleLayersNumber))
        self.modelSts = np.zeros((self.middleLayersNumber))
        self.middleLayerChanges = np.zeros((self.middleLayersNumber))
        self.lstmModelWeights = np.random.randn(1 + inLayerNumber + middleLayersNumber,4 * middleLayersNumber) / math.sqrt(inLayerNumber + middleLayersNumber)
        self.lstmModelWeights[0, :] = 0
        self.lstmModelChanges = np.zeros_like(self.lstmModelWeights)
        self.peepHoleConnectionWeights = np.ones((3, middleLayersNumber))
        self.peepHoleConnectionWeightsChgs = np.zeros_like(self.peepHoleConnectionWeights)

    def forwardPropogation(self, inValues):

        dim = inValues.shape[0]
        middleLayerBias = self.lstmModelWeights.shape[0]
        self.middleLayerLoad = np.zeros((dim, middleLayerBias))
        self.middleLayerAns = np.zeros((dim, self.middleLayersNumber))
        self.fomiAns = np.zeros((dim, self.middleLayersNumber * 4))
        self.fomiPrev = np.zeros((dim, self.middleLayersNumber * 4))
        self.presentSt = np.zeros((dim, self.middleLayersNumber))

        for index in range(dim):
            middleLayerPrevSt = self.middleLayerAns[index - 1, :] if (index > 0) else self.middleLayerSts
            self.middleLayerLoad[index, 0] = 1; self.middleLayerLoad[index, 1:1 + self.inLayerNumber] = inValues[index, :]
            beforeValues = self.presentSt[index - 1, :] if (index > 0) else self.modelSts
            self.middleLayerLoad[index, 1 + self.inLayerNumber:] = middleLayerPrevSt

            self.fomiPrev[index, :] = self.middleLayerLoad[index, :].dot(self.lstmModelWeights)
            self.fomiPrev[index, :self.middleLayersNumber] = self.fomiPrev[index, :self.middleLayersNumber] + np.multiply(beforeValues, self.peepHoleConnectionWeights[0, :])
            self.fomiPrev[index, self.middleLayersNumber:2 * self.middleLayersNumber] = self.fomiPrev[index,self.middleLayersNumber:2 * self.middleLayersNumber] + np.multiply(beforeValues, self.peepHoleConnectionWeights[1, :])
            self.fomiAns[index, 0:2 * self.middleLayersNumber] = 1.0 / (1.0 + np.exp(-self.fomiPrev[index, 0:2 * self.middleLayersNumber]))
            self.fomiAns[index, 3 * self.middleLayersNumber:] = np.tanh(self.fomiPrev[index, 3 * self.middleLayersNumber:])
            self.presentSt[index, :] = self.fomiAns[index,self.middleLayersNumber:2 * self.middleLayersNumber] * beforeValues + self.fomiAns[index,:self.middleLayersNumber] * self.fomiAns[index,3 * self.middleLayersNumber:]
            self.fomiPrev[index, 2 * self.middleLayersNumber:3 * self.middleLayersNumber] = self.fomiPrev[index,2 * self.middleLayersNumber:3 * self.middleLayersNumber] + np.multiply(self.presentSt[index, :], self.peepHoleConnectionWeights[2, :])
            self.fomiAns[index, 2 * self.middleLayersNumber:3 * self.middleLayersNumber] = 1.0 / (1.0 + np.exp(-self.fomiPrev[index, 2 * self.middleLayersNumber:3 * self.middleLayersNumber]))
            self.middleLayerAns[index, :] = self.fomiAns[index,2 * self.middleLayersNumber:3 * self.middleLayersNumber] * np.tanh(self.presentSt[index, :])

        self.modelSts = self.presentSt[index, :]
        self.middleLayerSts = self.middleLayerAns[index, :]


    def trainModel(self, learning_rate):
        for i, j, memory in zip([self.lstmModelWeights, self.peepHoleConnectionWeights],
                                      [self.lstmModelWeightsBack, self.peepHoleConnectionBack],
                                      [self.lstmModelChanges, self.peepHoleConnectionWeightsChgs]):
            memory += j * j
            i += -learning_rate * j / np.sqrt(memory + 1e-8)


    def resetValues(self):
        self.stateChanges = np.zeros((self.middleLayersNumber))
        self.middleLayerChanges = np.zeros((self.middleLayersNumber))
        self.modelSts = np.zeros((self.middleLayersNumber))
        self.middleLayerSts = np.zeros((self.middleLayersNumber))



    def backwardPropogation(self, backPropAnswer):
        self.lstmModelWeightsBack = np.zeros_like(self.lstmModelWeights)
        self.mofaGatebackNext = np.zeros_like(self.fomiAns);
        self.mofaGatebackPrev = np.zeros_like(self.fomiPrev)
        self.presentStBack = np.zeros_like(self.presentSt)
        self.peepHoleConnectionBack = np.zeros_like(self.peepHoleConnectionWeights);
        self.middleLayerNumberBck = np.zeros((self.middleLayersNumber));
        self.middleLayerAnsBack = backPropAnswer.copy(); self.middleLayerInBack = np.zeros_like(self.middleLayerLoad)
        dim = self.middleLayerLoad.shape[0]

        if self.stateChanges is not None: self.presentStBack[dim - 1] += self.stateChanges.copy()
        if self.middleLayerChanges is not None: self.middleLayerAnsBack[dim - 1] += self.middleLayerChanges.copy()
        for index in reversed(range(dim)):
            self.mofaGatebackNext[index,2 * self.middleLayersNumber:3 * self.middleLayersNumber] = self.middleLayerAnsBack[index, :] * np.tanh(self.presentSt[index, :])
            self.presentStBack[index, :] = self.presentStBack[index, :] + (self.middleLayerAnsBack[index, :] * self.fomiAns[index,2 * self.middleLayersNumber:3 * self.middleLayersNumber]) * (1 - np.tanh(self.presentSt[index, :] ** 2))
            if (index <= 0):
                self.mofaGatebackNext[index, self.middleLayersNumber:2 * self.middleLayersNumber] = self.presentStBack[index,:] * self.modelSts
                self.prevSt = self.fomiAns[index,self.middleLayersNumber:2 * self.middleLayersNumber] * self.presentStBack[index, :]
            else:
                self.mofaGatebackNext[index,self.middleLayersNumber:2 * self.middleLayersNumber] = self.presentStBack[index, :] * self.presentSt[index - 1,:]
                self.presentStBack[index - 1, :] = self.presentStBack[index - 1, :] +  self.fomiAns[index,self.middleLayersNumber:2 * self.middleLayersNumber] * self.presentStBack[index,:]

            self.mofaGatebackNext[index, :self.middleLayersNumber] = self.presentStBack[index,:] * self.fomiAns[index,3 * self.middleLayersNumber:]
            self.mofaGatebackNext[index, 3 * self.middleLayersNumber:] = self.presentStBack[index,:] * self.fomiAns[index,:self.middleLayersNumber]
            self.mofaGatebackPrev[index, 3 * self.middleLayersNumber:] = self.mofaGatebackNext[index,3 * self.middleLayersNumber:] * (1 - self.fomiAns[index,3 * self.middleLayersNumber:] ** 2)
            value = self.fomiAns[index, :3 * self.middleLayersNumber]
            self.mofaGatebackPrev[index, :3 * self.middleLayersNumber] = (value * (1 - value)) * self.mofaGatebackNext[index,:3 * self.middleLayersNumber]

            self.lstmModelWeightsBack = self.lstmModelWeightsBack + np.dot(self.middleLayerLoad[index:index + 1, :].T, self.mofaGatebackPrev[index:index + 1, :])
            self.middleLayerInBack[index, :] = self.mofaGatebackPrev[index, :].dot(self.lstmModelWeights.T)
            if index <= 0:
                self.peepHoleConnectionBack[0, :] = self.peepHoleConnectionBack[0, :] +  np.multiply(self.mofaGatebackPrev[index, :self.middleLayersNumber],self.modelSts)
                self.peepHoleConnectionBack[1, :] = self.peepHoleConnectionBack[1, :] + np.multiply(self.mofaGatebackPrev[index, self.middleLayersNumber:2 * self.middleLayersNumber], self.modelSts)
                self.peepHoleConnectionBack[2, :] = self.peepHoleConnectionBack[2, :] +  np.multiply(self.mofaGatebackPrev[index, 2 * self.middleLayersNumber:3 * self.middleLayersNumber],self.presentSt[index, :])
            else:
                self.peepHoleConnectionBack[0, :] = self.peepHoleConnectionBack[0, :] +  np.multiply(self.mofaGatebackPrev[index, :self.middleLayersNumber],self.presentSt[index - 1, :])
                self.peepHoleConnectionBack[1, :] = self.peepHoleConnectionBack[1, :] + np.multiply(self.mofaGatebackPrev[index, self.middleLayersNumber:2 * self.middleLayersNumber],self.presentSt[index - 1, :])
                self.peepHoleConnectionBack[2, :] = self.peepHoleConnectionBack[2, :] + np.multiply(self.mofaGatebackPrev[index, 2 * self.middleLayersNumber:3 * self.middleLayersNumber],self.presentSt[index, :])


            if (index > 0):
                self.middleLayerAnsBack[index - 1, :] = self.middleLayerAnsBack[index - 1, :] + self.middleLayerInBack[index, 1 + self.inLayerNumber:]
            else:
                self.middleLayerNumberBck = self.middleLayerNumberBck + self.middleLayerInBack[index, 1 + self.inLayerNumber:]


    def getHiddenLayerValues(self):
        return self.middleLayerAns