#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stores this repetitive data-generation process in a separate script.
"""
import numpy as np
# Set up the function to generate data (originally in dace.sim.r):
def rhogeodacecormat(geoD,rho,alpha=2):
     """
     A function to turn random uniform observations into a data array.
     
     Parameters
     ---------------------
     geoD: an interesting matrix
     rho: IDK
     alpha: specifies depth penalization?
     
     Returns
     -------
     R: Correlation matrix
     """
     # Rho can be an array or number in Python; we'll force it to be an array:
     rho = np.ones(1)*rho
     if (np.any(rho<0)):
          print("rho<0!"); exit()
     if (np.any(rho>1)):
          print("rho>1!"); exit()
     if (np.any(alpha<1) or np.any(alpha>2)):
          print("alpha out of bounds!"); exit()
     if(type(geoD) != np.ndarray):
          print("wrong format for distance matrices list!"); exit()
     # if(len(geoD) != len(rho)):
     #      print("rho vector doesn't match distance list"); exit()
     # ^Got rid of this warning because I'm doing my matrix alg. differently
     R = np.ones(shape=(geoD.shape[0], geoD.shape[0])) # Not sure about this line
     for i in range(len(rho)):
          R = R*rho[i]**(geoD**alpha)
          # ^ This is different notation than in R because my geoD array isn't a dataframe
     return(R)

def gen_data():
     # Generate response (data):
     np.random.seed(88)
     n = 10; rhotrue = 0.2; lambdatrue = 1
     # design = np.random.uniform(size=n).reshape(10,1)           # n x 1
     # For testing to compare to R's random seed, use this one:
     design = np.array([0.41050128,0.10273570,0.74104481,0.48007870,0.99051343,0.99954223,
               0.03247379,0.76020784,0.67713100,0.97679183]).reshape(n,1)
     l1 = np.subtract.outer(design[:,0],design[:,0])            # n x n   
     # ^ Not sure about this line, because the m1 is gone
     # ^ l1 is the same as l.dez in the R code, by the way
     R = rhogeodacecormat(l1,rhotrue)+1e-5*np.diag(np.ones(n))  # n x n
     L = np.linalg.cholesky(R)             # n x n
     # ^ For some weird reason, I don't need the transpose to make it match up with the L
     # matrix in the R code! Took me a while to figure out that one!
     # u=np.random.normal(size=R.shape[0]).reshape(1,10)          # n x 1
     # For testing to compare to R's random seed, use this one:
     u = np.array([0.1237811,0.1331487,-2.0407747,-1.2676089,0.6674839,-0.8014830,
                   0.9964860,1.3934232,-0.2291943,0.1707627]).reshape(n,1)
     y = np.matmul(L,u)                                         # n x 1
     return (design, y)