from plot_styling import * #Reads file and stuffs

#Constant frame rate is assumed.
dt = 0.03336 #For 30fps
"""
dt = 0.0416 for 24fps
dt = 0.04   for 25fps
"""

#Here center is a rectangle region of dimmensions half those of the stage located at the...
chist, yedg, xedg = np.histogram2d(y,x,bins=1,range=[[h/4, 3*h/4], [w/4, 3*w/4]])    #Center
whist, yedg, xedg = np.histogram2d(y,x,bins=1,range=[[0, h], [0, w]])                #Whole stage

#We asume no frame was skipped & tracked object was never out of sight.
Cthist = chist * dt
Rthist = whist * dt

print('Center time: {:.2f}s\nPeriphery time: {:.2f}s'.format(Cthist[0][0],Rthist[0][0]-Cthist[0][0]) )