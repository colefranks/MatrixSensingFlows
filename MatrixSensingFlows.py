
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



#### Operator Scaling Code 

#density matrix: mn x mn psd matrix
#target marginals: [m vector, n vector] (diagonals of target marginals)
#only works for reals for now
#only works for diagonal target marginals for now
#returns the pair of scalings and whether the desired precision was reached
#TODO: complex, non-diagonal, also output value of dual optimization function these are optimizing

def Sinkhorn(density_matrix, target_marginals, desired_precision=.001, max_iterations = 1000):
    left_target = target_marginals[0]
    right_target = target_marginals[1]
    left_dimension = np.shape(left_target)[0]
    right_dimension = np.shape(right_target)[0]
    converged = True
    error = math.inf
    iteration = 0
    current_scalings = [np.eye(left_dimension), np.eye(right_dimension)]
    while (error > desired_precision) and (iteration < max_iterations):
      print("iteration: ", iteration, "error: ", error)
      iteration+=1
      current_scalings,error = Sinkhorn_step(density_matrix, current_scalings, target_marginals)
    return current_scalings, error<desired_precision


def Sinkhorn_step(density_matrix, current_scalings, target_marginals):
  left_scaling = current_scalings[0]
  right_scaling = current_scalings[1]

  left_target = target_marginals[0]
  right_target = target_marginals[1]

  left_dimension = np.shape(left_target)[0]
  right_dimension = np.shape(right_target)[0]

  #fix right marginal
  marginal = trace_out_left(np.kron(left_scaling, np.eye(right_dimension)) @ density_matrix, left_dimension, right_dimension)
  right_tri_scaling = np.diag(np.sqrt(right_target))@ np.linalg.inv(scipy.linalg.cholesky(marginal, lower=True))
  right_scaling =  right_tri_scaling.T @ right_tri_scaling

  #fix left marginal
  marginal = trace_out_right(np.kron(np.eye(left_dimension), right_scaling) @ density_matrix, left_dimension, right_dimension)
  left_tri_scaling = np.diag(np.sqrt(left_target))@ np.linalg.inv(scipy.linalg.cholesky(marginal, lower=True))

  left_scaling =  left_tri_scaling.T @ left_tri_scaling

  #compute distance 

  tri_kron = np.kron(left_tri_scaling, right_tri_scaling)
  scaled_density_matrix = tri_kron @ density_matrix @ tri_kron.T
  error = 0
  error = error + np.linalg.norm(trace_out_right(scaled_density_matrix, left_dimension, right_dimension) - np.diag(left_target), 2)**2
  error = error + np.linalg.norm(trace_out_left(scaled_density_matrix, left_dimension, right_dimension) - np.diag(right_target), 2)**2


  return [left_scaling, right_scaling],error

#computes the dual objective of the thing

def dual_objective(density_matrix,current_scalings,target_marginals):

    left_scaling = current_scalings[0]
    right_scaling = current_scalings[1]

    left_target = target_marginals[0]
    right_target = target_marginals[1]

    left_dim = np.shape(left_target)[0]
    right_dim = np.shape(right_target)[0]

    #BorelObjective(X,b1,b2,P,Q,m,n):
    A = left_scaling.T @ left_scaling
    B = right_scaling.T @ right_scaling

    tra = np.trace(density_matrix @ np.kron(A,B))

    char1 = sum(left_target[i]*np.log(left_scaling[i,i]**2) for i in range(left_dim))
    char2 = sum(right_target[i]*np.log(right_scaling[i,i]**2) for i in range(right_dim))
    return tra - char1 - char2


#partial trace code

def trace_out_left(density_matrix, left_dimension, right_dimension):
      density_matrix = density_matrix.reshape([left_dimension, right_dimension, left_dimension, right_dimension])
      marginal = sum(density_matrix[i,:,i,:] for i in range(left_dimension))
      return marginal

def trace_out_right(density_matrix, left_dimension, right_dimension):
    density_matrix = density_matrix.reshape([left_dimension, right_dimension, left_dimension, right_dimension])
    marginal = sum(density_matrix[:,i,:,i] for i in range(right_dimension))
    return marginal

def random_spectrum(dim):
  spec = np.random.randn(dim)
  spec = spec*spec
  spec = spec/sum(spec)
  spec.sort()
  return np.flip(spec)


#### 




