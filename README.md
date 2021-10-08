# MatrixSensingFlows

This compares the nonconvex parameterization for matrix sensing explored in https://arxiv.org/abs/1705.09280 
in the special case of operator scaling with a gradient flow inspired by https://dl.acm.org/doi/10.1145/3188745.3188932. 

Also contains code implementing the main algorithm of the latter work, the "Sinkhorn" algorithm for operator scaling with arbitrary marginals.

Usage: to test flows on e.g. $5^2 x 5^2$ matrices, run CompareFlows(5)
