from plot_styling import *

bins = 20 #Number of divisions for heatmap

hist, yedg, xedg = np.histogram2d(y,x,bins=bins,range=[[0, h], [0, w]]) 
dt = data[0][1]-data[0][0]
thist = hist * dt #We asume no frame was skipped & tracked object was never out of sight

#Hexbin requires 1-D arrays
X,Y = np.meshgrid(np.linspace(0,w,bins),np.linspace(0,h,bins))
X = np.ndarray.flatten(X)
Y = np.ndarray.flatten(Y)
THIST = np.ndarray.flatten(thist)

fig, ax = plt.subplots()
hb = plt.hexbin(X,Y,C=THIST,gridsize=10,cmap=pur)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Time (s)')

#Comment this if you want the trayectory overlaid
#t = plt.plot(x,y,alpha=0.2,linewidth=2)

plt.show()