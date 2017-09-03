from plot_styling import *

fig, ax = plt.subplots()
pur = sns.cubehelix_palette(light=1,dark=0, as_cmap=True,reverse=True)
k = sns.kdeplot(x,y,shade=True,n_levels=1024,cmap=pur,xlim=(0,w),ylim=(0,h))

plt.xlim(0,w)
plt.ylim(0,h)
plt.show()