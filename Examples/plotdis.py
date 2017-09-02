import tkFileDialog as filedialog
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import Tkinter as tk
import pandas as pd
import numpy as np
import re

sns.set_style('dark')
mpl.rcParams['toolbar'] = 'None'

tk.Tk().withdraw()
file=filedialog.askopenfilename()

#Read dimensions from file name
dims = map(float, re.findall("\d+", file.split('/')[-1] ))
[w,h] = [dims[0],dims[1]]

data = pd.read_csv(file,header=None,dtype=None,delimiter=',')
data.columns = [ i for i in range(len(data.columns))]
x = np.array(data[1])
y = h - np.array(data[2])

fig, ax = plt.subplots()
pur = sns.cubehelix_palette(light=1,dark=0, as_cmap=True,reverse=True)
k = sns.kdeplot(x,y,shade=True,n_levels=1024,cmap=pur,xlim=(0,w),ylim=(0,h))

plt.xlim(0,w)
plt.ylim(0,h)
plt.show()