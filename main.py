#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:28:59 2018

@author: Qifan Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as optimize
import numpy.random as random
from Engine import Engine
from JohnRust import JohnRust



def main():
    pset = JohnRust(40)
    probIncrement, theta, replaceCost, numState = pset.getParamTrue
    stateArray = pset.getStateArray()
    
    maintainCostArray = pset.getMaintainCost(stateArray, theta)
    meanUtil = pset.getMeanUtil(replaceCost, maintainCostArray)
    
    ## solve continuation value function EV
    expectValue = pset.solveExpectValue(meanUtil = meanUtil, probIncrement=probIncrement)    
    pset.getExpectValuePlot(expectValue)
    
    probChoice = pset.getProbChoice(meanUtil, expectValue)
    
    ## data simulation 
    time = 1000
    numBus = 100
    stateSimulation, replaceSimulation = pset.dataSimulation(numBus,time, probChoice)
    pset.getProbIncrementHat(stateSimulation, replaceSimulation);
    pset.getOutputFile(time, numBus, stateSimulation, replaceSimulation)
    
    ## NFPA estimation
    pset.getParamNFPA(numState, stateSimulation, replaceSimulation)
        
    ## counterfactual analysis
    theta = 0.02
    path, demand, replaceSimulationArray, stateSimulationArray = pset.getDemand(theta, time, numBus)
    valueTotalGood = pset.getTotalValue(theta, demand, path, 
                                    replaceSimulationArray, stateSimulationArray)
    
    theta = 0.05
    path, demand, replaceSimulationArray, stateSimulationArray = pset.getDemand(theta, time, numBus)
    valueTotalBad = pset.getTotalValue(theta, demand, path, 
                                   replaceSimulationArray, stateSimulationArray)

if __name__ == '__main__':
   main()  
    
