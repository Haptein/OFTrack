from plot_styling import * #Reads file and stuffs

def circ_radii(b):
    return np.sqrt(b[0]**2+b[1]**2)

#Constant frame rate is assumed.
fps = input('Select video framerate:\n[0] 30fps\n[1] 25 fps\n[2] 24fps\n')
DT = [0.03336,0.04,0.0417]
dt = DT[fps]

center_type = input('Select center type:\n[0] Rectangular\n[1] Circular\n')

if center_type == 0:
    #Here center is a rectangle region of dimmensions half those of the stage located at the...
    chist, yedg, xedg = np.histogram2d(y,x,bins=1,range=[[h/4, 3*h/4], [w/4, 3*w/4]])    #Center
    whist, yedg, xedg = np.histogram2d(y,x,bins=1,range=[[0, h], [0, w]])                #Whole stage
    chist = chist[0][0]
    whist = whist[0][0]
else:
    #Here center is a circular region of radius half of the stage
    x = x - w/2
    y = y - h/2
    radii = map(circ_radii,np.transpose([x,y]))
    whist = len(radii)
    chist = sum([ r<(w/4) for r in radii])

#We asume no frame was skipped & tracked object was never out of sight.
Cthist = chist * dt
Wthist = whist * dt

print('Center time: {:.2f}s\nPeriphery time: {:.2f}s'.format(Cthist,Wthist-Cthist) )