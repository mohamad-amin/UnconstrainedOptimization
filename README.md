# UnconstrainedOptimization
Unconstrained optimization algorithms in python, line search and trust region methods.
There were two questions whose answers are available in:
* [Question 1 Jupyter Notebook](https://github.com/mohamad-amin/UnconstrainedOptimization/blob/master/question_1.ipynb)
* [Question 2 Jupyter Notebook](https://github.com/mohamad-amin/UnconstrainedOptimization/blob/master/question_2.ipynb)

The questions required two different kinds of unconstrained optimization algorithms:
* Line search algorithms
  * Steepest descent
      * Backtrack and cubic interpolation step length finder using wolfe and goldstein conditions
  * Quasi-Newton
    * Backtrack and cubic interpolation step length finder using wolfe and goldstein conditions
* Trust region algorithms
  * Dogleg step finder
  * Cauchy point step finder
  
There were also some needs for implementation of some linear algebric algorithms such as repairing positive semi definite matrices and ... which can be found in the [liang_utils.py](https://github.com/mohamad-amin/UnconstrainedOptimization/blob/master/algorithms/linalg_utils.py) file.
