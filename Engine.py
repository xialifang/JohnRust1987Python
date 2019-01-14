#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:16:02 2018

@author: Qifan Huang
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as optimize
import numpy.random as random

class Engine():
    def __init__(self, numState):
        self._probIncrement = [0.3, 0.5, 0.2]
        self._theta = 0.05
        self._replaceCost = 10
        self._beta = 0.99
        self._numState = numState
        self._directory = '/Users/mac/Desktop/'
    
    
    @property
    def getParamTrue(self):
        """
        Use: accessor of class JohnRust's field 
        """
        return  self._probIncrement, self._theta, self._replaceCost, self._numState       
    
       
    def solveExpectValue(self, meanUtil, probIncrement):
        """
        Use: compute expected value for each possible decision and each possible 
             state of the bus (mileage)
        Input: (1) numState: an int, number of states
               (2) meanUtil: a numState * 2 array 
        Return: evNew, a numState * 2 array, expected value for each state and choice       
        """
        ev = np.zeros((self._numState, 2))  ## expected value is a numState * 2 array
        evNew =  - np.ones((self._numState, 2))                      
        iterTime = 0 
        achieved = True    
        threshold = 0.000001 
        small = 10**(-323)    
        euler = 0.57721566490153286060651209008240243                            
                                              
        while np.abs(evNew - ev).max() > threshold:
            ev = evNew.copy()

            ## i = 0, NOT replace  i = 1, replace
            for i in range(2):
                probTransition = self.getProbTransition(probIncrement, i)
                ## sum over replacement choice
                evTemp = np.log(np.sum(np.exp(meanUtil + ev * self._beta), axis =1) + small)
                #evTemp = np.log(np.sum(np.exp(meanUtil + ev * self.__beta), axis =1) )
                evTemp = evTemp.reshape(1, self._numState)
                ## sum over state 
                evNew[:, i] = evTemp @ probTransition + euler 
                #evNew[:, i] = evTemp @ probTransition + meanUtil.copy()[:, i]
                           
            iterTime += 1
            if iterTime == 10000:
                achieved = False
                break
            
        if achieved == True:
            print("Convergence achieved in {} iterations".format(iterTime))
        else:
            print("CM could not converge! Mean difference = {:.6f}".format((evNew-ev).mean()))    
        return evNew                
                        
    
    def getStateArray(self):
        """
        Input: numState, an int
        Return: a numState * , array 
        """
        stateArray = np.arange(self._numState)
        return stateArray
        
        
    def getMaintainCost(self, stateArray, theta):
        """
        Use: calculate the cost of maintanence 
        Input: (1) stateArray: a numState * , array, total mileage
               (2) theta: a double, parameter in maintain cost function
        Return: a numState * , array, maintain cost                
        """
        maintainCostArray = stateArray * theta
        return maintainCostArray
    
    
    def getMeanUtil(self, replaceCost, maintainCostArray):
        """
        Use: calculate mean utility of current period
        Input: (1) numState: an int, number of states
               (2) replaceCost: a double, replacement cost
               (3) maintainCostArray: a numState * , array, maintain cost
        Return: a numState * 2 array, mean utility      
        """
        meanUtil = np.empty((self._numState, 2))
        ## first column, mean utility if NOT replacement
        meanUtil[:, 0] = - maintainCostArray 
        ## second column, mean utility if  replacement
        meanUtil[:, 1] = - replaceCost
        
        return meanUtil
        
                  
    def getProbTransition(self, probIncrementList, replace):
        """
        Use: calculate the probability of transition
        Input: (1) numState: an int, number of states
               (2) probIncrementList: a list with length 3, increment = 0, 1, 2
        Return: a numState * numState array        
        """
        ## row: future state    column: current state 
        probTransArray = np.zeros((self._numState, self._numState))
        
        if replace == 0:
            # column is current state 
            for i in range(self._numState):
                for j, probIncrement in enumerate(probIncrementList):
                    if i + j < self._numState - 1:
                        probTransArray[i+j][i] = probIncrement
                    elif i + j == self._numState - 1:
                        probTransArray[i+j][i] = sum(probIncrementList[j:])
                    else:
                        pass
        else:
            ## if replacement, next state must be 0 
            probTransArray = np.vstack((np.ones((1, self._numState)), 
                                        np.zeros((self._numState-1, self._numState))))                               
        return probTransArray                  
            
     
    def getProbChoice(self, meanUtil, expectValue):
        """
        Use: calculate the probability of replacement or not
        Input: (1) meanUtil: a numState * 2 array. 
               (2) expectValue: a numState * 2 array. 
        Return: a numState * 2 array. First column: prob of replacement   
                                      Second column: prob of NOT replacement
        """
        #small = 10**(-323)
        self.probChoiceArray = np.empty((self._numState, 2))
        #expSumUtil = np.exp(meanUtil + self._beta * expectValue).sum(1)[:, np.newaxis]
        #probChoiceArray = np.exp(meanUtil + self._beta * expectValue) / expSumUtil 
        for i in range(2):
            expSumUtil = np.exp(meanUtil + self._beta * expectValue 
                                - meanUtil[:, i][:, np.newaxis] - self._beta * expectValue[:, i][:, np.newaxis]).sum(1)#[:, np.newaxis]
            self.probChoiceArray[:, i] = 1 / expSumUtil
        #print(self.probChoiceArray)
        return self.probChoiceArray
    
    
    def getExpectValuePlot(self, expectValue):
        """
        Use: draw the plot of expected value over different choices 
        """
        plt.plot(range(self._numState), expectValue[:, 0], label = "EV if not replacement")
        plt.plot(range(self._numState), expectValue[:, 1], label = "EV if replacement")
        plt.legend()
        plt.xlabel("state")
        plt.ylabel("EV")
        plt.title("EV plot")                            
                    
    
    def dataSimulation(self, numBus,time, probChoice):
        """
        Use: simulate state and optimal decision.  
        Input:  (2) time: simulation periods 
                (3) probChoice: a numState * 2 array
        Return: (1) stateArray: numBus * time array (the first state is 0)
                (2) replaceArray: numBus * (time-1) array (don't calculate 
                   the last replacement choice )
                (3) a 5 * , array, which contains summary statistics of dataset 
        """
        stateArray = np.zeros((numBus, time))
        replaceArray = np.zeros((numBus, time - 1))        
        
        #if numState < time:
        #    raise Exception('the state space is not large enough')
        
        ## start to generate the data 
        for t in range(time - 1):
            
            ## randomly draw the mileage increment  
            mileIncrementRandom = stats.rv_discrete(values=([0, 1, 2], self._probIncrement))
            j = mileIncrementRandom.rvs(size=numBus)
            
            ## randomly draw the replacement choice 
            """
            replaceRandom = stats.rv_discrete(values=([0, 1], probChoice[t, :]))
            replace = replaceRandom.rvs(size=numBus)
            replaceArray[:, t] = replace 
            """
            """
            for i in range(numBus):
                replaceRandom = stats.rv_discrete(values=([0, 1],
                                              probChoice[int(stateArray[i, t]), :]))
                replace = replaceRandom.rvs(size=1)
                replaceArray[i, t] = replace
            """
            randomNum = random.uniform(0, 1, numBus)
            replaceArray[:, t] = randomNum < probChoice[stateArray[:, t].astype(int), 1]
            
            
            ## update the state for all buses 
            #stateArray[:, t+1] = stateArray[:, t] + j * (replaceArray[:, t] == 0)
            stateArray[:, t+1] = (stateArray[:, t] + j) * (replaceArray[:, t] == 0)
            stateArray[stateArray>= self._numState - 1] = self._numState - 1

        ## summarize the simulated data set         
        summary = np.zeros(5)
        summary[0] = np.mean(stateArray)
        summary[1] = np.std(stateArray)
        summary[2] = np.mean(replaceArray)
        summary[3] = np.std(replaceArray)
        summary[4] = np.corrcoef(np.mean(stateArray[:, :-1],0), 
                                 np.mean(replaceArray[:,:], 0))[0, 1]
        
        print(summary)
        
        plt.plot(range(time), np.mean(stateArray, axis = 0))
        plt.xlabel("period")
        plt.ylabel("state")
        plt.title("mean state over 100 buses (max state = " + str(self._numState) + ")")
        plt.show()
        
        plt.plot(range(time-1), np.mean(replaceArray, axis = 0))
        plt.xlabel("period")
        plt.ylabel("ratio of replacement")
        plt.title("ratio of replacement among 100 buses (max state = " + str(self._numState) + ")")
        plt.show()
    
        return stateArray,  replaceArray   
              
    
    def getOutputFile(self, time, numBus, stateSimulation, replaceSimulation):
        """
        create the output file for data simulation
        """
        filename = self.__directory + 'BusEngine_TZ.txt'
        data = np.zeros((numBus*(time - 1), 2))
        ## convert data form: bus 1 t1 bus 1 t2 bus 1 t3....bus 2 t1 bus 2 t2...
        data[:, 0] = stateSimulation[:, :-1].reshape((time-1)*numBus, )
        data[:, 1] = replaceSimulation.reshape((time-1)*numBus, )
        
        ## create the file 
        with open(filename, 'wb') as f:
            # write header
            np.savetxt(f, data, delimiter=",")
            
        return data     
    
    
