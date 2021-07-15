"""
Finds the row in the x-test that has the closest variable values to the x-train data,
and pairs their y and y-hat values for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt

def set_up_plot(fitp, x, y, points = 2000, var = [0, 1, 2, 3], day_range = 30):
     """
     Makes two arrays: y, and y-hat respectively, to plot. When plotted, they'll 
     show how close the fit is to predicting the correct y for a (predetermined) 
     set of variables.
     Parameters
     ----------
     fitp : dict
          fit predictions; will give us x_test and y_test arrays.
     x : numpy array
          x_train (data).
     y : numpy_array
          y_train (data responses).
     points : int, optional
          Determines how many points will be plotted - a random sample of
          length (points) of x is taken. The default is 2000.
     var : list, optional
          Specifies which variables to match   
     day_range : int, optional
          Allowance for the days variable (~1-1000) to not exactly match 
          the x_train day number in question. i.e. any entry within (day_offset) 
          days will be counted as a match. 25-40 seems to be a good value for this. 
          The default is 30.
     Returns
     -------
     y1, y2: y and yhat to plot
     """
     x_test = fitp['x_test'] # trials x p (8 for Walmart)
     y_test = fitp['mmean'] # trials x 1
     if(points > len(x)): points = len(x) # Capped by number of actual data points
     np.random.seed(88); count = 0 # Counting the loops where no x_train matches
     idxs = np.sort(np.random.choice(len(x), size = points, replace = False)) # Which points to actually use
     # print(idxs[0:25]) # A check
     y1 = np.empty(points); y2 = np.empty(points)
     for i in range(points): 
          # Find the matching x_train:
          good_idx = {}
          for v in var:
              if (v != 2): # Anything but the day variable:
                  good_idx[v] = np.where(x_test[:, v] == x[idxs[i], v])[0]   
              else: # Day variable
                  good_idx[v] = np.where(
                                np.abs(x_test[:, v] - x[idxs[i], v]) <= day_range)[0]
          if (len(var) == 1):
              good_idx_tot = good_idx[var[0]]
          elif (len(var) == 2):
              good_idx_tot = np.array(np.sort(list(set(
              good_idx[var[0]].intersection(good_idx[var[1]])))))
          elif (len(var) == 3):
              good_idx_tot = np.array(np.sort(list(set(set(
              good_idx[var[0]]).intersection(good_idx[var[1]])).intersection(good_idx[var[2]]))))
          elif (len(var) == 4):
              good_idx_tot = np.array(np.sort(list(set(set(set(
              good_idx[var[0]]).intersection(good_idx[var[1]]).intersection(good_idx[var[2]])).intersection(good_idx[var[3]])))))
          else: print("Variable comparisons list error; doesn't have 1-4 variables.")
          if (len(good_idx_tot) > 0):
              y1[i] = y[i] # y_train
              y2[i] = np.round(np.mean(y_test[good_idx_tot]), 2) # mean of y_tests that matched
          else:
              count = count + 1
     # print(good_idx_tot); print(len(good_idx[2])) # A check (on the last iteration of the loop)
     
     print('Number of x_train rows which were not perfectly matched in x_test:', count)
     # Delete the unfilled rows with 3 steps of masking:
     mask1 = (y2 > 10**(-9))
     y1 = y1[mask1]; y2 = y2[mask1]
     # ^ To get rid of the empties from the count thing
     mask2 = 0 < y2
     y1 = y1[mask2]; y2 = y2[mask2]
     mask3 = y2 < 1e+12
     y1 = y1[mask3]; y2 = y2[mask3]
     # ^ To get rid of a few wacky prediction values
     return(y1, y2) 



def pred_plot(y1, y2, title, fname, ms = 4, millions = True, lims = []):
    """
    Plots the output from the previous function (or from in-sample predictions).
    Parameters
    ----------
    y1 : numpy array
        AKA 'y': y_train array to plot.
    y2 : numpy array
        AKA 'y-hat': y_test array to plot.
    title : string
        Custom title of the plot
    fname : string
        File location to which to save the plot
    ms : int, optional
        markersize of points. The default is 4.
    millions : TYPE, optional
        If True, divide all y-values by a million. The default is True.
    lims : list, optional
        Specifies limits of the plot (if the defaults aren't good)   
    Returns
    -------
    None.
    """
    plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
    plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
    fig = plt.figure(figsize=(16,9)); ax = fig.add_subplot(111)
    if millions:
        ax.plot(y1/1000000, y2/1000000, 'ro', markersize = ms)
        ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
    else:
        ax.plot(y1, y2, 'ro', markersize = ms)
        ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
    if(lims == []):
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
    ax.plot(lims, lims, 'k-', linewidth=2); ax.set_xlim(lims); ax.set_ylim(lims);
    ax.set_title(title)
    plt.savefig(fname)
    
    
    
def vartiv_plot(fitv, title, fname, labels):
    """
    Plots the output of vartivity() in a boxplot: shows the proportion of
    tree rules attributed to each variable.
    Parameters
    ----------
    fitv : dictionary (fit object)
        Contains vartivity results. Could also be changed to manually
        inputting mvdraws from a file in the future.
    title : string
        Custom title of the plot.
    fname : string
        File location to which to save the plot.
    labels : list
        Labels for each boxplot on the x-axis.
    Returns
    -------
    None
    """
    plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
    plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
    fig = plt.figure(figsize=(16,16)); ax = fig.add_subplot(111)
    ax.boxplot(fitv['vdraws'], labels = labels)
    ax.set_ylabel("Proportion of Tree Rules")
    ax.set_title(title)
    plt.savefig(fname)
    
    
    
def sobol_plot(fits, title, fname, labels, p = ['msi', 'mtsi'], ij = False):
    """
    Plots the output of sobol() in a boxplot: shows the proportion of
    one-way, two-way, and/or total sobol sensitivities attributed to each variable.
    Parameters
    ----------
    fits : dictionary (fit object)
        Contains sobol results. Could also be changed to manually
        inputting msi, msij, and/or mtsi from a file in the future.
    title : string
        Custom title of the plot.
    fname : string
        File location to which to save the plot.
    labels : list
        Labels for the x-axis for the non-sij plot - they correspond with variable names.
    p : list, optional
        Lists which sobol results to plot side-by-side: can be 1 or 2 of
        the aforementioned results. The default is msi and mtsi.
    ij : boolean, optional
        If True, plot the msij's (which have a different number of pairs to plot).
        The default is False.
    Returns
    -------
    None
    """
    plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
    plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
    fig = plt.figure(figsize=(16,9)); ax = fig.add_subplot(111)
    c = ['r', 'b']
    if (ij == False):
        sd = ['msi_sd', 'tsi_sd']; x = np.arange(len(fits['msi'])) + 1
        for i in range(len(p)):  
            ax.errorbar(x, fits[p[i]], yerr = fits[sd[i]], color = c[i], label = p[i])
        ax.set_xticks(x); ax.set_xticklabels(labels)
    else:
        x = np.arange(len(fits['msij'])) + 1
        ax.errorbar(x, fits['msij'], yerr = fits['sij_sd'], color = 'g', label = 'msij')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("Pair of variables")
    ax.set_ylabel("Proportion of Sobol Sensitivity")
    ax.set_title(title)
    ax.legend(loc = "upper right", prop={'size': 25})
    plt.savefig(fname)
