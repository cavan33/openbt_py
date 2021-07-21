"""
Branin Function example for Python openbt package
"""
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

from openbt.openbt import OPENBT, load
m = OPENBT(model = "bart", tc = 4, modelname = "branin")
fit = m.fit(x, y)

# Calculate in-sample predictions
fitp = m.predict(x, tc = 4)

# Make a simple plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,9)); ax = fig.add_subplot(111)
ax.plot(y, fitp['mmean'], 'ro')
ax.set_xlabel("Observed"); ax.set_ylabel("Fitted")
ax.axline([0, 0], [1, 1])

#--------------------------------------------------------------------------------------------
# Save fitted MODEL object (not the estimator object, m) as test.obt in the working directory
m.save(fit, "test", est = False)
# Load fitted model object (AKA fit object) to a new object
fit2 = load("test", est = False)

# We can also save/load the fit ESTIMATOR object by specifying est = True in save()/load().
# The estimator object has all our settings and properties, but not fit results. 
# This is similar to scikit-learn saving/loading its estimators.
m.save("test_fit_est", est = True)
m2 = load("test_fit_est", est = True)
# If you wish, you can see that m2 (the loaded estimator object) can perform fits:
# fit3 = m2.fit(x, y)
# m2 can perform predictions, too:
# fitp2 = m2.predict(x, tc = 4)
#--------------------------------------------------------------------------------------------

# Calculate variable activity information
fitv = m.vartivity()
print(fitv['mvdraws'])

# Calculate Sobol indices
fits = m.sobol(cmdopt = 'MPI', tc = 4)
print(fits['msi'])
print(fits['mtsi'])
print(fits['msij'])
