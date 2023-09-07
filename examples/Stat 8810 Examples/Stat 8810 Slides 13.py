"""
Derived from this example: http://www.matthewpratola.com/teaching/stat8810-fall-2017/,
Slides 13 Code. It uses the BayesTree package, but it's similar to using OpenBT.

This script replicates the OpenBT fit behavior using Python. I took functions from
Zoltan Puha's repo, but made a new config file (openbt) which was tailored
to how I wanted to set some more parameters.
"""
import os
os.chdir("/home/clark/Documents/OpenBT/OpenBT_package2/src/openbt") # Will be different for your files
import numpy as np
import matplotlib.pyplot as plt
import sys
from openbt.openbt import OPENBT # You can now reference the class just by typing OPENBT
sys.path.append("PyScripts/Stat 8810 Examples/Functions") # Might be different for your filesystem
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
ntree = 1 # Default = 1
ntreeh = 1 # Default = 1
npred_arr = 25
# For plotting:
npreds = 100 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'PyScripts/Plots/' # Might be different for your filesystem

#---------------------------------------------------------------------------------------
def fit_pipeline(design, y, model, ndpost, nadapt, nskip, power, base, tc, numcut, ntree,
                 ntreeh, k, overallsd, overallnu, npreds, fig, path, fname):
     m = OPENBT(model=model, ndpost=ndpost, nadapt = nadapt, nskip=nskip, power=power,
                base=base, tc=tc, numcut=numcut, ntree=ntree, ntreeh=ntreeh, k=k,
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
# Fit BART, with many different settings, to see how the fit responds:
(plot1, fit1, fitp1, m1) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=ntree,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-1.png')
plt.clf()

# Try m=10 trees
m=10
(plot2, fit2, fitp2, m2) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-2.png')
plt.clf()

# Try m=20 trees
m=20
(plot3, fit3, fitp3, m3) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-3.png')
plt.clf()

# Try m=100 trees
m=100
(plot4, fit4, fitp4, m4) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-4.png')
plt.clf()

# Try m=200 trees, the recommended default
m=200
(plot5, fit5, fitp5, m5) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-5.png')
plt.clf()

# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
(plot6, fit6, fitp6, m6) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-6.png')
plt.clf()

# For all other runs here, it's OK to ignore q, since openbt currently doesn't
# have a setting for it.
# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=3, q=.99
nu=3
(plot7, fit7, fitp7, m7) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=nu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-7.png')
plt.clf()

# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=2, q=.99
nu=2
(plot8, fit8, fitp8, m8) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=nu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-8.png')
plt.clf()

# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
(plot9, fit9, fitp9, m9) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=nu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-9.png')
plt.clf()

# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000
(plot10, fit10, fitp10, m10) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=m,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=nu,
               npreds=npreds, fig=fig, path=path, fname='Slides 13-10.png')
plt.clf()

#----------------------------------------------------------------------------------
# Example - the CO2 Plume data from Assignment 3
# Fit the model
co2plume = np.loadtxt('PyScripts/newco2plume.txt', skiprows=1)
# Kinda cheated, and made the tricky .dat file into a .txt file using R
x = co2plume[:,0:2] # Not including the 3rd column
y = co2plume[:,2]
preds = np.array([(x, y) for x in range(npred_arr) for y in range(npred_arr)])/(npred_arr-1)
preds = np.flip(preds,1) # flipped columns to match the preds in the R code

shat = np.std(y, ddof = 1)
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000

# Do this one manually, since it's a different setup than what I wrote the
# function for:
m11 = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit11 = m11.fit(x,y)
fitp11 = m11.predict(preds)

# Plot CO2plume posterior samples of sigma
fig = plt.figure(figsize=(10,5.5))
ax = fig.add_subplot(111)
ax.plot(m11.sdraws, color='black', linewidth=0.15)
ax.set_xlabel('Iteration'); ax.set_ylabel('$\sigma$')
ax.set_title('sdraws during Python CO2Plume fitp');
plt.savefig(f'{path}co2plume_sdraws.png')
plt.clf()

# Now plot the original points and the fit on top:
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib qt5
# ^ Comment this line out if not running in iPython; supposed to help the plot show correctly in some cases
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['axes.labelsize'] = 12;
plt.rcParams['xtick.labelsize'] = 10; plt.rcParams['ytick.labelsize'] = 10;
ax.scatter(co2plume[:,0], co2plume[:,1], co2plume[:,2], color='black')
ax.set_xlabel('Stack_inerts'); ax.set_ylabel('Time'); ax.set_zlabel('CO2')
plt.savefig(f'{path}co2plume_orig.png')

a = np.arange(0, 1.0001, 1/(npred_arr-1)); b = a;
A, B = np.meshgrid(a, b)
ax.plot_surface(A, B, m11.mmean.reshape(npred_arr,npred_arr), color='black')
ax.set_xlabel('Stack_inerts'); ax.set_ylabel('Time'); ax.set_zlabel('CO2')
plt.savefig(f'{path}co2plume_fit.png')

# Add the uncertainties (keep the surface from above, too):
ax.plot_surface(A, B, (m11.mmean + 1.96 * m11.smean).reshape(npred_arr,npred_arr), color='green')
ax.plot_surface(A, B, (m11.mmean - 1.96 * m11.smean).reshape(npred_arr,npred_arr), color='green')
plt.savefig(f'{path}co2plume_fitp.png')
