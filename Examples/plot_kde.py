from plot_styling import *

fig, ax = plt.subplots()
k = sns.kdeplot(x,y,shade=True,n_levels=512,cmap=pur,xlim=(0,w),ylim=(0,h))

plt.xlim(0,w)
plt.ylim(0,h)
plt.show()