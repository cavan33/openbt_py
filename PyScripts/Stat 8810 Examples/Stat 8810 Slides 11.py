"""
Derived from this example: http://www.matthewpratola.com/teaching/stat8810-fall-2017/,
Slides 11 Code. It uses the BayesTree package, but it's similar to using OpenBT.

This script replicates the OpenBT fit behavior using Python. I took functions from
Zoltan Puha's repo, but made a new config file (openbt.py) which was tailored
to how I wanted to set some more parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from openbt import OPENBT # Should be in the working directory (no path.append needed)
sys.path.append("PyScripts/Stat 8810 Examples/Functions")
from gen_data8810 import *
# Example (Our usual GP realization) originally using BayesTree, 
# now written in Python with openbt-python.
design, y = gen_data()

# Now, set up the fit:
# Set values to be used in the fitting/predicting:
# Variance and mean priors
overallsd = 0.1 # (AKA shat) # in reality , it's lambdatrue^2?
# lower shat --> more fitting, I think
overallnu = 3
k = 2 # lower k --> more fitting, I think

# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
nc = 100 # (AKA numcut); Default = 100

# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100
nadapt = 1000 # Default = 1000
tc = 4 # Default = 2, but we usually use 4
ntree = 1 # (AKA m); Default = 1
ntreeh = 1 # Default = 1

# For plotting:
npreds = 100 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'PyScripts/Plots/' # Might be different for your filesystem
fname = 'Slides11.png'
#---------------------------------------------------------------------------------------
def fit_pipeline(design, y, model, ndpost, nskip, power, base, tc, numcut, ntree,
                 ntreeh, k, overallsd, overallnu, npreds, fig, path, fname):
     m = OPENBT(model=model, ndpost=ndpost, nskip=nskip, power=power, base=base,
                tc=tc, numcut=numcut, ntree=ntree, ntreeh=ntreeh, k=k,
                overallsd=overallsd, overallnu=overallnu)
     fit = m.fit(design,y)
     preds = np.arange(0, (1 + 1/npreds), 1/(npreds-1)).reshape(npreds, 1)
     fitp = m.predict(preds)

     # Plot predictions:
     ax = fig.add_subplot(111)
     ax.plot(design, y, 'ro')
     ax.set_title(f'Predicted mean response +/- 2 s.d., ntree = {ntree}')
     # ^ Technically the +/- will be 1.96 SD
     ax.set_xlabel('Observed'); ax.set_ylabel('Fitted'); ax.set_ylim(-1.5, 1)
     # Plot the central line: Overall mean predictor
     ax.plot(preds, m.mmean, 'b-', linewidth=2)

     # Make the full plot with the gray lines: (mmeans and smeans are now returned by m.predict()!)
     ax.plot(preds, m.mmean - 1.96 * m.smean, color='black', linewidth=0.8)
     ax.plot(preds, m.mmean + 1.96 * m.smean, color='black', linewidth=0.8)
     if (ndpost < npreds):
          print('Number of posterior draws (ndpost) are less than the number of', 
                'x-values to predict. This is not recommended.')
          npreds = ndpost
     for i in range(npreds):
          ax.plot(preds, m.mdraws[i, :],color="gray", linewidth=1, alpha = 0.20)
     plt.savefig(f'{path}{fname}')
     return((fig, fit, fitp, m)) # Returns the plot, fit results, and the instance of the class

#---------------------------------------------------------------------------------------
# Fit BART
(plot, fit, fitp, m) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=ntree,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname=fname)
# plt.clf()

# Testing the load() and save() functions:
m.save_fit(fit, "PyScripts/Results/Stat_8810/save_fit11")
fit_2 = m.load_fit("PyScripts/Results/Stat_8810/save_fit11")

fitv = m.vartivity()
# fits (sobol function) # doesn't work for this example because p = 1
