import tkFileDialog as filedialog
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import Tkinter as tk
import pandas as pd
import numpy as np
import re

#Styling
sns.set_style('dark')
mpl.rcParams['toolbar'] = 'None'
pur = sns.cubehelix_palette(light=1,dark=0, as_cmap=True,reverse=True)

#File selection
tk.Tk().withdraw()
file=filedialog.askopenfilename()

#Read dimensions from file name
dims = map(float, re.findall("\d+", file.split('/')[-1] ))
[w,h] = [dims[0],dims[1]]

#Load file data
data = pd.read_csv(file,header=None,dtype=None,delimiter=',')
data.columns = [ i for i in range(len(data.columns))]

t = np.array(data[0])
x = np.array(data[1])
y = h - np.array(data[2])
#Vertical axis coordinates in video start from above.