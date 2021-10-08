
import tensorflow as tf
import numpy as np
import scipy as scipy
from scipy import optimize
import cvxpy as cp
import matplotlib.pyplot as plt
from numpy import matrix
from sympy import *
import math
import itertools


#exploring whether the gradient flows from matrix sensing and operator scaling converge to the same thing

#gradient flow for matrix sensing

def GradientFlow(U_start, As, ys, numSteps, learningRate=1e-4,verbose = False):
    As, ys = tf.constant(As), tf.constant(ys)
    U = tf.Variable(U_start)
    Matrices = []
    
    for i in range(numSteps):
        DoGradientStep(U, As, ys, learningRate)
        X = U @ tf.transpose(U)
        Y = np.array(X)
        Matrices.append(Y)
        if verbose and i%1000==0:
          vals, vecs  = np.linalg.eigh(Y)

          print("error: ", np.array(SquaredError(X, As, ys)))
          
    return U, Matrices

@tf.function
def DoGradientStep(U, As, ys, learningRate):
    dydU = Gradient(U, As, ys)
    U.assign_sub(learningRate*dydU)

@tf.function
def Gradient(U, As, ys):
    """ The goal is that <As[i], U @ U.T> = ys[i] for each index i. """
    with tf.GradientTape() as g:
        g.watch(U)
        y = ParametrizedSquaredError(U, As, ys)
    return g.gradient(y, U)

@tf.function
def ParametrizedSquaredError(U, As, ys):
    return SquaredError(U @ tf.transpose(U), As, ys)

def SquaredError(X, As, ys):
    """ This is the quantity that the gradient flow tries to minimize. """
    As_T = tf.transpose(As, perm=[0,2,1])
    innerProducts = tf.linalg.trace(As_T @ X)
    errors = ys - innerProducts
    return tf.reduce_sum(errors**2)

###### defining gradient flow for parameterizing X = UU^T for u upper triangular

#defining a kronecker product, since tensorflow doesn't provide one
@tf.function
def tf_kron(A,B,m,n):
  A = tf.reshape(A,(m*m,1))
  B = tf.reshape(B,(n*n,1))
  kron = A @ tf.transpose(B)
  kron = tf.reshape(kron, (m,m,n,n))
  kron = tf.transpose(kron, perm = [0,2,1,3])
  kron = tf.reshape(kron,(m*n,m*n))
  #return tf.reshape(A@tf.transpose(B),(m*n,m*n))
  return kron

def BorelObjective(X,b1,b2,P,Q,m,n):
    A = b1 @ tf.transpose(b1)
    B = b2 @  tf.transpose(b2)
    tra = tf.linalg.trace(X @ tf_kron(A,B,m,n))
    char1 = sum(P[i,i]*tf.math.log(b1[i,i]**2) for i in range(m))
    char2 = sum(Q[i,i]*tf.math.log(b2[i,i]**2) for i in range(n))
    return tra - char1 - char2

#gradient of that objective wrt b1,b2
@tf.function
def BorelGradient(X,b1,b2,P,Q,m,n):
    with tf.GradientTape() as g:
        g.watch(b1)
        y = BorelObjective(X,b1,b2,P,Q,m,n)
        b1grad = g.gradient(y,b1)
    with tf.GradientTape() as g:
        g.watch(b2)
        y = BorelObjective(X,b1,b2,P,Q,m,n)
        b2grad = g.gradient(y,b2)

    # keep it lower triangular (the above simply assumes b1,b2 are any matrices)
    tri = np.ones([m,m])
    for j in range(m):
      for i in range(j):
        tri[i,j] = 0
    b1grad = tf.math.multiply(b1grad,tri)

    tri = np.ones([n,n])
    for j in range(n):
      for i in range(j):
        tri[i,j] = 0
    b2grad = tf.math.multiply(b2grad,tri)

    return b1grad, b2grad

@tf.function
def BorelDoGradientStep(X,b1,b2,P,Q,m,n, learningRate):
    db1,db2 = BorelGradient(X,b1,b2,P,Q,m,n)

    b1.assign_sub(learningRate*db1)
    b2.assign_sub(learningRate*db2)

def BorelGradientFlow(X, P,Q, numSteps=10000, learningRate=1e-4,verbose = False):
    m = np.shape(P)[0]
    n = np.shape(Q)[0]
    P, Q = tf.constant(P,dtype = "float64"), tf.constant(Q,dtype="float64")
    b1 = tf.Variable(np.eye(m),dtype = "float64")
    b2 = tf.Variable(np.eye(n),dtype = "float64")

    
    for i in range(numSteps):
        if verbose and i%1000==0:
          gA, gB = BorelGradient(X,b1,b2,P,Q,m,n)
          print("step: ", i, "error: ",np.linalg.norm(gA) + np.linalg.norm(gB), "objective", np.array(BorelObjective(X,b1,b2,P,Q,m,n)))
        BorelDoGradientStep(X,b1,b2,P,Q,m,n, learningRate)
    tens = tf_kron(b1,b2,m,n)
    #outputs the final state. 
    return np.array(tens@X@tf.transpose(tens))

######

#matrix sensing measurements for operator scaling

def OperatorScalingMeasurements(n):

    # operator scaling A's
    A_list = []
    n =5
    ys = []
    #marg1 = np.array([.55,.45])
    marg1 = np.random.randn(n)
    marg1 = marg1*marg1
    marg1 = np.flip(np.sort(marg1))
    marg1 = marg1/sum(marg1)
    marg2 = np.random.randn(n)
    marg2 = marg2*marg2
    marg2 = np.flip(np.sort(marg2))
    marg2 = marg2/sum(marg2)



    #first the off-diagonals
    for i in range(n):
     for j in range(i+1):
        elt = np.zeros([n,n])
        elt[i,j]=1

        A_list.append(np.kron(elt,np.eye(n)))
        A_list.append(np.kron(np.eye(n),elt))

        if not i==j:
            ys.append(0)
            ys.append(0)
        else:
            ys.append(marg1[i])
            ys.append(marg2[i])

    return A_list, ys, marg1, marg2


def CompareFlows(n):
    #test whether the matrix sensing and operator scaling flows are the same 

    A_list, ys, marg1, marg2 = OperatorScalingMeasurements(n)

    U_start = np.random.randn(n**2, n**2)
    X = U_start @ U_start.T
    #P = np.diag([1.,4.])
    #Q = np.diag([1.,2.,2.])
    P = np.diag(marg1)
    Q = np.diag(marg2)
    #Xfinal = OpGradientFlow(X,P,Q,numSteps = 10000,learningRate =1e-3,verbose = True)
    Xfinal = BorelGradientFlow(1e-10*X,P,Q,numSteps = 100000,learningRate =.1,verbose = True)

    U_new, Entries = GradientFlow(1e-5*U_start, A_list, ys, 20000, learningRate=1e-3, verbose=True)

    U_new = np.array(U_new)
    X_new = U_new @ U_new.T
    print("norm of difference: ", np.linalg.norm(Xfinal-X_new))

    np.linalg.norm(X_new - Xfinal)
    vals, vecs = np.linalg.eigh(Xfinal)
    vals1, vecs = np.linalg.eigh(X_new)

    #comparing eigenvalues
    plt.plot(vals)
    plt.plot(vals1)





