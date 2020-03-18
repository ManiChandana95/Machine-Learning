import numpy as np
import matplotlib.pyplot as plt
from LSTMModel import LSTMModel
import sklearn.preprocessing as preprocessing
import pandas as pd
import statistics

class RNNLSTM:
    def __init__(self):
        print("Our project predicts the stock data using RNN (LSTM) ")
        print("Starting the process")

    def train(self,x_trData, y_trData, x_tsData, y_tsData):

        tAV = list(); tPV = list(); tL = list(); indexValue = 0; lossIndex = 0
        outW = np.random.randn(1, 20) / np.sqrt(1)
        lstmRnnObject = LSTMModel(1, 20)
        inValues = np.zeros((19, 1))
        tValues = np.zeros((19, 1))
        changedOWeights = np.zeros_like(outW)

        for runs in range(2000):
            if indexValue + 19 + 1 >= len(x_trData):
                indexValue = 0
                lstmRnnObject.resetValues()
            inValues[:, 0] = x_trData[indexValue, :]
            tValues[:, 0] = y_trData[indexValue]
            tAV.append(np.mean(np.square(tValues)))
            lstmRnnObject.forwardPropogation(inValues)
            hLOutput = lstmRnnObject.getHiddenLayerValues()
            output = hLOutput.dot(outW.T)
            differenceValue = output - tValues
            tPV.append(np.mean(output))
            changedBackWeights = (differenceValue).T.dot(hLOutput)
            loss = np.mean(np.square(output - tValues))
            tL.append(loss)
            backPropAnswer = (differenceValue).dot(outW)
            lstmRnnObject.backwardPropogation(backPropAnswer)
            lstmRnnObject.trainModel(0.005)

            for i, j, memory in zip([outW],
                                      [changedBackWeights],
                                      [changedOWeights]):
                memory += j * j
                i += -0.005 * j / np.sqrt(memory + 1e-8)

            print(lossIndex,loss)
            indexValue = indexValue+1; lossIndex = lossIndex+1

        print("Total Training Loss: ", str(np.sum(tL)))
        print("Training Accuracy for our system: ", 100 - (abs(statistics.mean([tPV_i - tAV_i for tPV_i, tAV_i in zip(tPV, tAV)]))) * 100)

        self.test(x_tsData,y_tsData,outW,lstmRnnObject)

    def ConvertDataToPredict(self,df):
        numpy_array = df.to_numpy()
        values = list()
        for num in range(len(numpy_array) - 20):
            values.append(numpy_array[num: num + 20])

        values = np.array(values)
        trainDataSize = int(np.round(25 / 100 * values.shape[0]));
        modelDataSize = values.shape[0] - trainDataSize

        x_trData = values[:modelDataSize, :-1, :]
        y_trData = values[:modelDataSize, -1, :]
        x_tsData = values[modelDataSize:, :-1, :]
        y_tsData = values[modelDataSize:, -1, :]

        return x_trData[:, :, 0], y_trData[:, 0], x_tsData[:, :, 0], y_tsData[:, 0]

    def plotData(self,dataFrameStock):
        plt.subplot(1, 2, 1);
        plt.plot(dataFrameStock.Open.values, color='green', label='Stock Start Rate')
        plt.plot(dataFrameStock.Close.values, color='red', label='Stock Close Rate')
        plt.plot(dataFrameStock.Low.values, color='blue', label='Stock Low Rate')
        plt.plot(dataFrameStock.High.values, color='black', label='Stock High Rate')
        plt.title('Stock Prices for 50 Years')
        plt.xlabel('Days')
        plt.ylabel('Rate')
        plt.legend(loc='best')
        plt.subplot(1, 2, 2);
        plt.plot(dataFrameStock.Volume.values, color='blue', label='Volume Sold')
        plt.title('Stock Volume')
        plt.xlabel('Days')
        plt.ylabel('Volume')
        plt.legend(loc='best')

    def test(self,x_tsData, y_tsData,outW,lstmRnnObject):
        tAV = list()
        tPV = list()
        loss = list()
        outW = np.random.randn(1, 20) / np.sqrt(1)
        inValues = np.zeros((19, 1))
        tValues = np.zeros((19, 1))
        indexValue = 0
        for index in range(300):
            if indexValue + 20 >= len(x_tsData):
                indexValue = 0
                lstmRnnObject.resetValues()
            inValues[:, 0] = x_tsData[indexValue, :]
            tValues[:, 0] = y_tsData[indexValue]
            tAV.append(np.mean(np.square(tValues)))
            lstmRnnObject.forwardPropogation(inValues)
            insideOutput = lstmRnnObject.getHiddenLayerValues()
            output = insideOutput.dot(outW.T)
            tPV.append(np.mean(output))
            temploss = np.mean(np.square(output - tValues))
            loss.append(temploss)
            indexValue = indexValue+1

        print("Total Testing Loss: ", str(np.sum(loss))); print("Testing Accuracy for our system: ", 100 - (abs(statistics.mean(
            [tPV_i - tAV_i for tPV_i, tAV_i in zip(tPV, tAV)]))) * 100)
        plt.figure(2)
        plt.plot(tAV, color='red', label='actual')
        plt.plot(tPV, color='green', label='predicted')
        plt.title('Predicted vs Original Values of Stock Data')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='best');
        plt.show()


if __name__ == "__main__":

    rnnlstm = RNNLSTM()

    #Plot the data
    df = pd.read_csv("https://raw.githubusercontent.com/Abhishek1004/ML/master/%5EIXIC%20(1).csv", index_col=0, header = [0], sep = ',')
    rnnlstm.plotData(df)

    #Drop the unused columns
    df.drop(['Volume'], 1, inplace=True)
    df.drop(['Adj Close'], 1, inplace=True)

    #Preprocessing Step
    df.Open = preprocessing.MaxAbsScaler().fit_transform(df.Open.values.reshape(-1, 1))
    df.High = preprocessing.MaxAbsScaler().fit_transform(df.High.values.reshape(-1, 1))
    df.Low = preprocessing.MaxAbsScaler().fit_transform(df.Low.values.reshape(-1, 1))
    df.Close = preprocessing.MaxAbsScaler().fit_transform(df.Close.values.reshape(-1, 1))

    #Getting the data for Prediction
    x_trData, y_trData, x_tsData, y_tsData = rnnlstm.ConvertDataToPredict(df)

    #Train the model and get the accuracy for training and testing accuracy
    rnnlstm.train(x_trData,y_trData,x_tsData,y_tsData)