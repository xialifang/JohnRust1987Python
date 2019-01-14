#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:50:26 2018

@author: Qifan Huang 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as optimize
import numpy.random as random
from Engine import Engine


class JohnRust(Engine):    
    def __init__(self, numState):
        Engine.__init__(self, numState)
    
    
    def getProbIncrementHat(self, stateSimulation, replaceSimulation):
         ## step 1 outer loop: estimate transition probability 
        stateDiff = np.diff(stateSimulation, axis = -1)
        
        ## only keep those don't choose replacement last period 
        stateCondDiff = stateDiff[np.where(replaceSimulation == 0)]        
        ## probIncrement estimation
        self.probIncrementHat = []
        for j in [0, 1, 2]:
            self.probIncrementHat.append(np.sum(stateCondDiff == j) / len(stateCondDiff))
        
        
    def getParamNFPA(self, numState, stateSimulation, replaceSimulation):      
        ## step 2 inner loop: estimate transition probability
        x0 = [0.1 for i in range(2)]
        bounds = [(0, None) for i in range(2)]
        self.fitted = optimize.fmin_l_bfgs_b(self.logLikelihood,  x0=x0, 
                               args=(numState , stateSimulation , replaceSimulation 
                               , self.probIncrementHat), approx_grad=True, bounds=bounds)
        print(self.fitted[0])
        print(self.fitted[1])
        
        
    def logLikelihood(self, params, numState , stateSimulation , replaceSimulation 
                          , probIncrementHat):
        theta = params[0]
        replaceCost = params[1]
        
        numBus = replaceSimulation.shape[0]
        time = replaceSimulation.shape[1]
        
        stateArray = self.getStateArray()
        maintainCostArray = self.getMaintainCost(stateArray, theta)
        meanUtil = self.getMeanUtil(replaceCost, maintainCostArray)
        expectValue = self.solveExpectValue(meanUtil = meanUtil, probIncrement=probIncrementHat)
        probChoice = self.getProbChoice(meanUtil, expectValue)
                            
        logLikelihood = 0
        
        for i in range(numBus):
            for j in range(time-1):
                state = int(stateSimulation[i, j])
                replace = int(replaceSimulation[i, j])
                p = probChoice[state, replace] 
                logLikelihood += np.log(p)

        return -logLikelihood  
    
      
    def getDemand(self, theta, time, numBus):
        stateArray = self.getStateArray()
        maintainCostArray = self.getMaintainCost(stateArray, theta)
        path = np.arange(0, 30, 5)
        length = len(path)
        demand = np.zeros((length, time - 1))
        replaceSimulationArray = np.zeros((length, numBus, time - 1))
        stateSimulationArray = np.zeros((length, numBus, time - 1))
   
        for i, replaceCost in enumerate(path):
            meanUtil = self.getMeanUtil(replaceCost, maintainCostArray)
            expectValue = self.solveExpectValue(meanUtil, self._probIncrement)
            probChoice = self.getProbChoice(meanUtil, expectValue)
            stateSimulation, replaceSimulation = self.dataSimulation(numBus,time, probChoice)
            demand[i, :] = np.sum(replaceSimulation, axis = 0)
            replaceSimulationArray[i, :, :] = replaceSimulation
            stateSimulationArray[i, :, :] = stateSimulation[:, :-1]
        
        for i, replaceCost in enumerate(path):
            #plt.plot(np.arange(time - 1), demand[i, :], label = "RC = " + str(replaceCost))
            plt.plot(np.arange(100), demand[i, 0:100], label = "RC = " + str(replaceCost))
            plt.legend()
            plt.xlabel("period")
            plt.ylabel("demand")
        plt.title("Demand plot when theta = " + str(theta))  
        plt.show()   
        
        return path, demand, replaceSimulationArray, stateSimulationArray
        
    
    def getTotalValue(self, theta, demand, path, 
                  replaceSimulationArray, stateSimulationArray):        
        if theta == 0.05:
            mc = 10
        if theta == 0.02:
            mc = 20
        
        numBus = replaceSimulationArray.shape[1]
        time = replaceSimulationArray.shape[2]
                
        replaceSimulation = replaceSimulationArray[path == mc, :, :].reshape(numBus, time)
        stateSimulation = stateSimulationArray[path == mc, :, :].reshape(numBus, time)
            
        utilFlow =  (replaceSimulation==1) *  (- self._replaceCost) 
                       +  (replaceSimulation==0) * (- stateSimulation * self._theta)
                        
        ## discounted sum of flow utility                
        utilDiscount = np.zeros(utilFlow.shape[0])
        mcTotal = 0
        
        for t in range(utilFlow.shape[1]):
            utilDiscount +=  utilFlow[:, t] * self._beta**(t)              
            mcTotal += mc * demand[path == mc, t].sum() * self._beta**(t) 
            
        valueTotal = utilDiscount.sum() - mcTotal
        print(valueTotal)
        return valueTotal

        
        
        
        
            
        


