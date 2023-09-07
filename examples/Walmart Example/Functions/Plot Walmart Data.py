"""
Import the Walmart data and plot it to visualize what it's like.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("PyScripts/Walmart Example") # Might be different for your filesystem
from Construct_Walmart_Data import *


# Load in the data:
(x, y, x_pd, y_pd) = get_walmart_data()

# Basic plots of this:
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'PyScripts/Plots/Walmart/' # Might be different for your filesystem   
  
fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
ax.scatter(x[:, 2], y/1000000, s = 8)
ax.set_title('Walmart Weekly Store Sales')
ax.set_xlabel('Days since 01-01-2010'); ax.set_ylabel('Weekly Store Sales (Million $)')
plt.savefig(f'{path}ExamplePlot1.png')

# For individual stores only:
x1 = x[x[:, 0] == 1]; y1 = y[x[:, 0] == 1]
x2 = x[x[:, 0] == 2]; y2 = y[x[:, 0] == 2]
x3 = x[x[:, 0] == 22]; y3 = y[x[:, 0] == 22]
x4 = x[x[:, 0] == 36]; y4 = y[x[:, 0] == 36]
fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
ax.plot(x1[:, 2], y1/1000000, 'r.-', linewidth=2)
ax.plot(x2[:, 2], y2/1000000, 'g.-', linewidth=2)
ax.plot(x3[:, 2], y3/1000000, 'b.-', linewidth=2)
ax.plot(x4[:, 2], y4/1000000, '.k-', linewidth=2)
ax.set_title('Walmart Weekly Store Sales for 4 Stores')
ax.set_xlabel('Days since 01-01-2010'); ax.set_ylabel('Weekly Store Sales (Million $)')
plt.savefig(f'{path}ExamplePlot2.png')

fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
ax.scatter(x[:, 7], y/1000000, s = 9)
ax.set_title('Walmart Weekly Store Sales vs. Area Unemployment')
ax.set_xlabel('Unemployment Rate (%)'); ax.set_ylabel('Weekly Store Sales (Million $)')
plt.savefig(f'{path}ExamplePlot3.png')

fig = plt.figure(figsize=(10,5.5)); ax = fig.add_subplot(111)
ax.scatter(x[:, 6], y/1000000, s = 9)
ax.set_title('Walmart Weekly Store Sales vs. CPI')
ax.set_xlabel('Consumer Price Index'); ax.set_ylabel('Weekly Store Sales (Million $)')
plt.savefig(f'{path}ExamplePlot4.png')
