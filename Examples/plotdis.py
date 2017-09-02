import tkFileDialog as filedialog
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import Tkinter as tk
import pandas as pd
import numpy as np

sns.set_style('dark')

mpl.rcParams['toolbar'] = 'None'

tk.Tk().withdraw()
file=filedialog.askopenfilename()
[w,h] = [33,21]
data = pd.read_csv(file,header=None,dtype=None,delimiter=',')
data.columns = [ i for i in range(len(data.columns))]
x = np.array(data[1])
y = h - np.array(data[2])

fig, ax = plt.subplots()

pur = sns.cubehelix_palette(light=1,dark=0, as_cmap=True,reverse=True)
k = sns.kdeplot(x,y,shade=True,n_levels=256,cmap=pur,xlim=(0,w),ylim=(0,h))


#w = ax.plot([0,0,w,w,0],[0,h,h,0,0],linewidth=10,color='black')
plt.xlim(0,w)
plt.ylim(0,h)


plt.show()
