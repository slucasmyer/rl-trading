# an example of how to plot in real time.
# might be useful for exploring training sessions since they take forever to run.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()# Turn on interactive mode

fig, ax = plt.subplots() # Create a figure and axis

x, y = [], []
line, = ax.plot(x, y, 'r-')

for i in range(100):
    x.append(i)
    y.append(np.random.rand())
    
    line.set_xdata(x)
    line.set_ydata(y)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.ioff() # Turn off interactive mode