"""
Import the Walmart data and run sobol/fitv on it to observe and plot their differences
"""
import numpy as np; import random
import matplotlib.pyplot as plt
import sys
from openbt import OPENBT # Should be in the working directory (no path.append needed)
sys.path.append("PyScripts/Walmart Example/Functions")
from Construct_Walmart_Data import *
from summarize_output import *
from walmart_pred_plot import *

# Load in the data (8 x variables, after I edited it):
(x, y, x_pd, y_pd) = get_walmart_data()

# Settings:
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 2000 # (AKA ndpost); Default = 1000
burn = 2000 # (AKA nskip); Default = 100
nadapt = 2000 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 5 # Default = 2
shat = np.std(y, ddof = 1)
m=200
k=1
nu=1
nc=2000

""" # For non-in-sample predictions:
npred_arr = 40000 # Rows of the preds x grid
preds_list = []; np.random.seed(88)
# Categorical variables:
for col in [0, 1, 2]:
     preds_list.append(np.random.randint(np.min(x[:, col]), np.max(x[:, col])+1, size = npred_arr))
# Separate, weighted one for holiday flag, since it's more zeros than ones:
preds_list.append(random.choices([0, 1], weights = (1-np.mean(x[:, 3]), np.mean(x[:, 3])), k = npred_arr))    
# Continuous variables:
for col in [4, 5, 6, 7]:
     preds_list.append(np.random.uniform(np.min(x[:, col]), np.max(x[:, col])+2.2e-16, size = npred_arr))
preds = np.asarray(preds_list).T # This is supposedly faster than having it be a np array the whole time
# print(preds.nbytes / 1000000) # To view storage size in MB
"""

preds = x # For in-sample
m = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit = m.fit(x,y)
fitp = m.predict(preds)
# summarize_fitp(fitp)

# Vartivity:
fitv = m.vartivity()
# summarize_fitv(fitv)

# Sobol:
fits = m.sobol(cmdopt = 'MPI', tc = tc)
# summarize_fits(fits)

# Save fit objects:
fpath1 = 'PyScripts/Results/Walmart Example/'
save_fit_obj(fit, f'{fpath1}fit_result.txt', objtype = 'fit')
save_fit_obj(fitp, f'{fpath1}fitp_result.txt', objtype = 'fitp')
save_fit_obj(fitv, f'{fpath1}fitv_result.txt', objtype = 'fitv')
save_fit_obj(fits, f'{fpath1}fits_result.txt', objtype = 'fits')


# Plot y vs yhat plots:
""" # For non-in-sample:
ys, yhats = set_up_plot(fitp, x, y, points = len(x), var = [0, 1, 2, 3], day_range = 30)
pred_plot(ys, yhats, 'BART y vs. $\hat(y)$, Full Settings, all 4 Variables',
  'PyScripts/Plots/Walmart/y-yhat1',
  ms = 1.5, millions = True, lims = [0.0, 3.1])
"""

yhats = fitp['mdraws'].mean(axis = 0)
pred_plot(y, yhats, 'In-Sample BART y vs. $\hat(y)$, Full Settings',
  'PyScripts/Plots/Walmart/y-yhat-in-sample1',
  ms = 1.5, millions = True, lims = [0.0, 3.1])

# Plot boxplots to compare mvdraws (vartivity) to sobol results
vartiv_plot(fitv, 'Vartivity mvdraws for In-Sample BART, Full Settings',
  'PyScripts/Plots/Walmart/mvdraws1',
  labels = list(x_pd.columns))

sobol_plot(fits, 'Sobol Sensitivities for In-Sample BART, Full Settings',
  'PyScripts/Plots/Walmart/sobols1',
  labels = list(x_pd.columns))

labs = ['1,2', '1,3', '1,4', '1,5', '1,6', '1,7', '1,8', '2,3', '2,4', '2,5', 
        '2,6', '2,7', '2,8', '3,4', '3,5', '3,6', '3,7', '3,8', '4,5', '4,6', 
        '4,7', '4,8', '5,6', '5,7', '5,8', '6,7', '6,8', '7,8']
sobol_plot(fits, 'Sobol Sensitivities (S_i,j) for In-Sample BART, Full Settings',
  'PyScripts/Plots/Walmart/sobols2', 
  labels = labs, ij = True)
