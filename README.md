# pyopenbt
This Python package is the Python interface for Dr. Matthew Pratola's [OpenBT project](https://bitbucket.org/mpratola/openbt/wiki/Home). Currently, its only module is openbt, which contains the OPENBT class. This class allows the user to create fit objects in a scikit-learn style.

[![Build](https://github.com/cavan33/openbt_py/actions/workflows/python-package.yml/badge.svg)](https://github.com/cavan33/openbt_py/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/pyopenbt.svg)](https://badge.fury.io/py/pyopenbt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyopenbt/badges/version.svg)](https://anaconda.org/conda-forge/pyopenbt)

### About:  
OpenBT is a flexible and extensible C++ framework for implementing Bayesian regression tree models. Currently a number of models and inference tools are available for use in the released code with additional models/tools under development. The code makes use of MPI for parallel computing. Apart from this package, an R interface is provided via the ROpenbt package to demonstrate use of the software.

### How to utilize this package (and its module and class):  
1. Install the package from the command line by typing:  
`$ python -m pip install pyopenbt`.   
2. In Python3 (or a Python script), import the OPENBT class from the openbt module by typing:  
`from pyopenbt.openbt import OPENBT`.  
This gives Python access to the OPENBT class. Typing  
`from pyopenbt.openbt import *`  
or  
`from pyopenbt import openbt`  
would also work, but for the former, the obt_load() function is loaded unnecesarily (unless you wish to use that function, of course). For the latter, the class would be referred to as `pyopenbt.OPENBT`, not simply OPENBT.  
3. To utilize the OPENBT class/functions in Python 3 to conduct and interpret fits: create a fit object such as  
`m = OPENBT(model = "bart", ...)`.  
The fit object is an instance of the class. Here's an example of running a functions from the class:  
`fitp = m.predict(preds)`
4. See example scripts (in the "examples" folder), showing the usage of the OPENBT class on data, to this package. 

### Example:  
To start, let's create a test function. A popular one is the [Branin](https://www.sfu.ca/~ssurjano/branin.html) function:
```
# Test Branin function, rescaled
def braninsc (xx):
    x1 = xx[0]
    x2 = xx[1]
    
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    
    import math
    term1 = x2bar - 5.1*x1bar**2/(4*math.pi**2) + 5*x1bar/math.pi - 6
    term2 = (10 - 10/(8*math.pi)) * math.cos(x1bar)
    
    y = (term1**2 + term2 - 44.81) / 51.95
    return(y)


# Simulate branin data for testing
import numpy as np
np.random.seed(99)
n = 500
p = 2
x = np.random.uniform(size=n*p).reshape(n,p)
y = np.zeros(n)
for i in range(n):
    y[i] = braninsc(x[i,])
```
Note that the x and y data is a numpy array - this is the intended format. Now we can load the openbt package and fit a BART model. Here we set the model type as model="bart" which ensures we fit a homoscedastic BART model. The number of MPI threads to use is specified as tc=4. For a list of all optional parameters, see `m._dict__` (after creating m) or `help(OPENBT)`.

```
from pyopenbt.openbt import OPENBT, obt_load
m = OPENBT(model = "bart", tc = 4, modelname = "branin")
fit = m.fit(x, y)
```
Next we can construct predictions and make a simple plot comparing our predictions to the training data. Here, we are calculating the in-sample predictions since we passed the same x array to the predict() function.
```
# Calculate in-sample predictions
fitp = m.predict(x, tc = 4)

# Make a simple plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,9)); ax = fig.add_subplot(111)
ax.plot(y, fitp['mmean'], 'ro')
ax.set_xlabel("Observed"); ax.set_ylabel("Fitted")
ax.axline([0, 0], [1, 1])
```
To save the model, use OPENBT's obt_save() function. Similarly, load the model using obt_load(). Because the posterior can be large in sample-based models such as these, the fitted model is saved in a compressed file format with the extension .obt. Additionally, the estimator object can be saved and loaded (see below).
```
#--------------------------------------------------------------------------------------------
# Save fitted MODEL object (not the estimator object, m) as test.obt in the working directory
m.obt_save(fit, "test", est = False)
# Load fitted model object (AKA fit object) to a new object
fit2 = obt_load("test", est = False)

# We can also save/load the fit ESTIMATOR object by specifying est = True in obt_save()/load().
# The estimator object has all our settings and properties, but not fit results. 
# This is similar to scikit-learn saving/loading its estimators.
m.obt_save("test_fit_est", est = True)
m2 = obt_load("test_fit_est", est = True)
#--------------------------------------------------------------------------------------------
```
The standard variable activity information, calculated as the proportion of splitting rules involving each variable, can be computed using OPENBT's vartivity() function.
```
# Calculate variable activity information
fitv = m.vartivity()
print(fitv['mvdraws'])
```
A more accurate alternative is to calculate the Sobol indices.
```
# Calculate Sobol indices
fits = m.sobol(cmdopt = 'MPI', tc = 4)
print(fits['msi'])
print(fits['mtsi'])
print(fits['msij'])
```
Again, for more examples of using OpenBT, explore the examples folder in the [Github repo](https://github.com/cavan33/openbt_py) .

### See Also:  
[Github "Homepage" for this package](https://github.com/cavan33/openbt_py)  
PyPI [Package Home](https://pypi.org/project/openbt/)  

### Contributions
All contributions are welcome. You can help this project be better by reporting issues, bugs, 
or forking the repo and creating a pull request.

------------------------------------------------------------------------------

### License
The package is licensed under the BSD 3-Clause License. A copy of the
[license](LICENSE.txt) can be found along with the code.