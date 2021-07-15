"""
Functions to more easily save the results of fit predictions, sobol sensitivities, and/or vartivity metrics to text files.
"""
import numpy as np
def summarize_fitp(fitp):
     print(fitp['mdraws'][0, 0:5]); print(np.mean(fitp['mdraws']))
     print(fitp['sdraws'][0, 0:5]); print(np.mean(fitp['sdraws']))
     # print(fitp['mmean'][0:5]); print(np.mean(fitp['mmean']))
     # print(fitp['smean'][0:5]); print(np.mean(fitp['smean']))
     print(fitp['msd'][0:5]); print(np.mean(fitp['msd']))
     print(fitp['ssd'][0:5]); print(np.mean(fitp['ssd']))
     print(np.mean(fitp['m_5'])); print(np.mean(fitp['m_lower'])); print(np.mean(fitp['m_upper']))
     print(np.mean(fitp['s_5'])); print(np.mean(fitp['s_lower'])); print(np.mean(fitp['s_upper']))
def summarize_fitv(fitv):
     print(fitv['vdraws'][0:3, :]); print(np.mean(fitv['vdraws']))
     print(fitv['vdrawsh'][0:3, :]); print(np.mean(fitv['vdrawsh']))
     print(fitv['mvdraws']); print(fitv['mvdrawsh'])
     print(fitv['vdraws_sd']); print(fitv['vdrawsh_sd']); print(fitv['vdraws_5'])
     print(fitv['vdrawsh_5']); print(fitv['vdraws_lower']); print(fitv['vdraws_upper'])
     print(fitv['vdrawsh_lower']); print(fitv['vdrawsh_upper'])
def summarize_fits(fits):
     print(np.mean(fits['vidraws'])); print(np.mean(fits['vijdraws']))
     print(np.mean(fits['tvidraws'])); print(np.mean(fits['vdraws']))
     print(np.mean(fits['sidraws'])); print(np.mean(fits['sijdraws']))
     print(np.mean(fits['tsidraws']))
     # print(fits['msi']); print(fits['msi_sd']); print(fits['si_5'])
     # print(fits['si_lower']); print(fits['si_upper']); print(fits['msij'])
     # print(fits['sij_sd']); print(fits['sij_5']); print(fits['sij_lower'])
     # print(fits['sij_upper']); print(fits['mtsi']); print(fits['tsi_sd'])
     # print(fits['tsi_5']); print(fits['tsi_lower']); print(fits['tsi_upper'])
     
def save_fit_obj(fit, fname, objtype):
     from pathlib import Path
     to_save = dict(fit) # A "shallow copy"
     configfile = Path(fname)
     with configfile.open("w") as f:
          for cat in to_save:
               f.write(str(cat)+": \n")
               f.write(str(to_save[cat])+"\n")
          if objtype == 'fitp': # (Nothing needed for objtype == 'fit')
               f.write("Mean(mdraws): \n"); f.write(str(np.mean(fit['mdraws']))+"\n")
               f.write("Mean(sdraws): \n"); f.write(str(np.mean(fit['sdraws']))+"\n")
               f.write("Mean(msd): \n"); f.write(str(np.mean(fit['msd']))+"\n")
               f.write("Mean(ssd): \n"); f.write(str(np.mean(fit['ssd']))+"\n")
               f.write("Mean(m_lower): \n"); f.write(str(np.mean(fit['m_lower']))+"\n")
               f.write("Mean(m_5): \n"); f.write(str(np.mean(fit['m_5']))+"\n")
               f.write("Mean(m_upper): \n"); f.write(str(np.mean(fit['m_upper']))+"\n")
               f.write("Mean(s_lower): \n"); f.write(str(np.mean(fit['s_lower']))+"\n")
               f.write("Mean(s_5): \n"); f.write(str(np.mean(fit['s_5']))+"\n")
               f.write("Mean(s_upper): \n"); f.write(str(np.mean(fit['s_upper']))+"\n")
          elif objtype == 'fits':
               f.write("Mean(vidraws): \n"); f.write(str(np.mean(fit['vidraws']))+"\n")
               f.write("Mean(vijdraws): \n"); f.write(str(np.mean(fit['vijdraws']))+"\n")
               f.write("Mean(tvidraws): \n"); f.write(str(np.mean(fit['tvidraws']))+"\n")
               f.write("Mean(vdraws): \n"); f.write(str(np.mean(fit['vdraws']))+"\n")
               f.write("Mean(sidraws): \n"); f.write(str(np.mean(fit['sidraws']))+"\n")
               f.write("Mean(sijdraws): \n"); f.write(str(np.mean(fit['sijdraws']))+"\n")
               f.write("Mean(tsidraws): \n"); f.write(str(np.mean(fit['tsidraws']))+"\n")
               f.write("Mean(msij): \n"); f.write(str(np.mean(fit['msij']))+"\n")
               f.write("Mean(sij_sd): \n"); f.write(str(np.mean(fit['sij_sd']))+"\n")
               f.write("Mean(sij_5): \n"); f.write(str(np.mean(fit['sij_5']))+"\n")
               f.write("Mean(sij_lower): \n"); f.write(str(np.mean(fit['sij_lower']))+"\n")
               f.write("Mean(sij_upper): \n"); f.write(str(np.mean(fit['sij_upper']))+"\n")
               f.write("Mean(so_draws): \n"); f.write(str(np.mean(fit['so_draws']))+"\n")
               
  
def save_fits_old(fits, fname):
     from pathlib import Path
     to_save = dict(fits) # A "shallow copy"
     for cat in ('vidraws', 'vijdraws', 'tvidraws', 'vdraws', 'sidraws', 'sijdraws', 'tsidraws'):
          del to_save[cat]
     configfile = Path(fname)
     with configfile.open("w") as tfile:
          for cat in to_save:
               tfile.write(str(cat)+": \n")
               tfile.write(str(to_save[cat])+"\n")
