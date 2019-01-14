# JohnRust1987Python

| INTRODUCTION  | README |
| :-----: | :------: |
| **Author:** | Qifan Huang |
| **Date:** | December 11, 2018 |
| **Email** | Qifan@uw.edu |
| **License:**| JohnRust1987Python is released under the MIT license.|

&nbsp;
## Description
JohnRust1987Python is a python implementation of Nested Fixed Point Approach (NFPA) in John Rust's paper "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher".

&nbsp;
## Structure
1. main.py: 
   - compute and plot continuation value function EV
   - data simulation based on model in John Rust (1987) 
   - parameter estimation using NFPA
   - counterfactual analysis
2. Engine.py:
   - parent class Engine, which contains methods of data simulation, solving EV (contraction mapping) and calculating choice probability 
3. JohnRust.py:
   - child class JohnRust, which implements Nested Fixed Point Approach and counterfactual analysis
   
&nbsp;
## Reference
Rust J. Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher[J]. Econometrica: Journal of the Econometric Society, 1987: 999-1033.
