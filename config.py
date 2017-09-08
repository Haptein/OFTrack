#!/usr/bin/python2.7
import tkFileDialog as filedialog
import Tkinter as tk
import os, cv2, sys
import numpy as np
import argparse

#[Dimx,Dimy,CC,ratio,FPS,out_res,AvF_thresh]
DEFAULTS = np.array([33,21,0,3,30,2,0,170])

#Resolution options
#RES = ["1920x1080","1280x720","960x540", "640x360"]
RES = ["3840x1080","2560x720","1920x540","1280x360"]
#Color Config
CC = ["Dark Animal / Clear Surface", "Clear Animal / Dark Surface"] 
#Scaling options
SC = ["1/1","1/2","2/3","1/3","1/4"]
#Video Formats
EXT = ['avi','mp4','mkv']
#Config file name
cfile = 'oft.conf'

def reload():
    global DEFAULTS
    try:
        conf_data = np.genfromtxt(cfile,delimiter=",",dtype=None)
        print("Config file loaded successfully.")
    except:
        print("Couldn't load config file properly.")
        np.savetxt(cfile,DEFAULTS.reshape(1,len(DEFAULTS)),delimiter=',',fmt='%s')
        conf_data = DEFAULTS
        print("New config file created.")
    return conf_data


class getparams(tk.Tk):

    def __init__(self):
        
        tk.Tk.__init__(self)
        self.wm_title("OF Tracker Configuration")
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
        self.RES = tk.StringVar()
        self.EXT = tk.StringVar()
        self.CC = tk.StringVar()
        self.SC = tk.StringVar()

        #Labels
        self.ST1 = tk.Label(self,text='Experimental Settings',font=(20))
        self.ST2 = tk.Label(self,text='Display',font=(20))
        self.ST3 = tk.Label(self,text='Video Output',font=(20))
        self.Ldimx = tk.Label(self,text='Length(cm)')
        self.Ldimy = tk.Label(self,text='Width(cm)')
        self.Lcc = tk.Label(self,text='Color Configuration')
        self.Lsc = tk.Label(self,text='Video Scaling')
        self.Lext = tk.Label(self,text='Video Format')
        self.Lfps = tk.Label(self,text='FPS')
        self.Lres = tk.Label(self,text='Resolution')
        self.thresh = tk.Label(self,text='Threshold: '+str(AvF_thresh))

        #Inputs
        self.dimx = tk.Entry(self,width=15)
        self.dimy = tk.Entry(self,width=15)
        self.cc = tk.OptionMenu(self, self.CC, "Dark Animal / Clear Surface", "Clear Animal / Dark Surface")
        self.sc = tk.OptionMenu(self, self.SC, "1/1","1/2","2/3","1/3","1/4")
        self.fps = tk.Entry(self,width=15)
        #self.res = tk.OptionMenu(self, self.RES, "1920x1080","1280x720","960x540", "640x360")
        self.res = tk.OptionMenu(self, self.RES, "3840x1080","2560x720","1920x540","1280x360")
        self.ext = tk.OptionMenu(self, self.EXT, "avi","mp4","mkv")######
        self.cal = tk.Button(self, text="Calibrate", command=self.on_cal)
        self.save = tk.Button(self, text="Save Settings", command=self.on_save)
        self.exit = tk.Button(self, text="Exit", command=self.on_exit)
        
        #conf_data:[Dimx,Dimy,CC,ratio,FPS,out_res,AvF_thresh]
        #Set vars
        self.dimx.insert(0,conf_data[0])
        self.dimy.insert(0,conf_data[1])
        self.CC.set(CC[conf_data[2]])
        self.SC.set(SC[conf_data[3]])
        self.fps.insert(0,conf_data[4])
        self.RES.set(RES[conf_data[5]])
        self.EXT.set(EXT[conf_data[6]])

        #Placement
        self.ST1.grid(row=0, column=0, pady=10)
        self.Ldimx.grid(row=1, column=0)
        self.dimx.grid(row=1, column=1)
        self.Ldimy.grid(row=2, column=0)
        self.dimy.grid(row=2, column=1)
        self.Lcc.grid(row=3, column=0)
        self.cc.grid(row=3, column=1)
        self.cal.grid(row=4, column=0)
        self.thresh.grid(row=4, column=1)
        self.ST2.grid(row=5, column=0, pady=10)
        self.Lsc.grid(row=6, column=0)
        self.sc.grid(row=6, column=1)
        self.ST3.grid(row=7, column=0, pady=10)
        self.Lfps.grid(row=8, column=0)
        self.fps.grid(row=8, column=1)
        self.Lres.grid(row=9, column=0)
        self.res.grid(row=9, column=1)
        self.Lext.grid(row=10, column=0)
        self.ext.grid(row=10, column=1)
    
        self.save.grid(row=11, column=0, pady=10)
        self.exit.grid(row=11, column=1, pady=10)
        self.lift()

    #conf_data:[Dimx,Dimy,CC,sc,FPS,res,ext,AvF_thresh]    
    def on_save(self):
        global DEFAULTS, RES, CC, SC, AvF_thresh, conf_data
        try:
            dimx = int(self.dimx.get())
            dimy = int(self.dimy.get())
            cc = [i for i in range(len(CC)) if CC[i]==self.CC.get()][0]
            sc = [i for i in range(len(SC)) if SC[i]==self.SC.get()][0]
            fps = int(self.fps.get())
            res = [i for i in range(len(RES)) if RES[i]==self.RES.get()][0]
            ext = [i for i in range(len(EXT)) if EXT[i]==self.EXT.get()][0]
            Settings = np.array([dimx,dimy,cc,sc,fps,res,ext,AvF_thresh]).reshape(1,len(DEFAULTS))
            np.savetxt(cfile,Settings,delimiter=',',fmt='%s')
            conf_data = np.array([dimx,dimy,cc,sc,fps,res,ext,AvF_thresh])
        except:
            print('saving failed.')
        
        
    def on_exit(self):
        self.destroy()
        sys.exit()

    def on_cal(self):
        global conf_data,SC,AvF_thresh
        [DimX,DimY,cc,RA,FPS,res,ext,THRESHOLD_ANIMAL_VS_FLOOR] = conf_data
        RA = SC[RA]
        RA = RA.split('/')
        ratio = float(RA[0])/float(RA[1])

        if args.live:
            filename = args.live
            name = 'Live'
        else:
            tk.Tk().withdraw()
            filename=filedialog.askopenfilename()
            name = os.path.splitext(filename)[0]

        if not filename:  return

        cap = cv2.VideoCapture(filename)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(h*ratio)
        w = int(w*ratio)

        ret, frame = cap.read()
        nullframes = 0
        while not frame.any():
            ret, frame = cap.read()
            nullframes += 1
            if nullframes > 50:
                break
        
        if ret:
            print('Frame found.')
        else: 
            print('No frames found.')
            return

        perspectiveMatrix = floorCrop(filename, conf_data, args)
        
        if  len(perspectiveMatrix[name])==0:
            print('No perspective.')
            return

        kernelSize = (25, 25)

        #Creating a trackbar really needs a function input
        def nothing(x):
            pass

        cv2.namedWindow('Calibration')
        cv2.createTrackbar('Threshold','Calibration',AvF_thresh,255,nothing)

        cc = conf_data[2]
        frame_counter = 2
        while(1):

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            THRESHOLD_ANIMAL_VS_FLOOR = cv2.getTrackbarPos('Threshold','Calibration')

            #Press spacebar to set threshold value
            if k == 32:
                AvF_thresh = THRESHOLD_ANIMAL_VS_FLOOR
                self.thresh['text'] = 'Threshold: ' + str(AvF_thresh)
                
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame,(w,h))
            frameColor = frame.copy()

            if not cc:
                frame = cv2.bitwise_not(frame)

            #Apply mask if provided
            if (mask is not None) and mask.shape == frame.shape:
                frameColor = frameColor * mask
                frame = frame * mask

            #Text and text shadows
            txtoff = 1
            cv2.putText(frameColor, 'Set threshold so that the animal/object',
                (10+txtoff,20+txtoff), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['black'])
            cv2.putText(frameColor, 'to track is the only visible blob.',
                (10+txtoff,50+txtoff), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['black'])
            cv2.putText(frameColor, 'Press Space to set a new threshold value, Esc to exit.',
                (10+txtoff,h-30+txtoff), cv2.FONT_HERSHEY_DUPLEX, 0.6, BGR_COLOR['black'])

            cv2.putText(frameColor, 'Set threshold so that the animal/object',
                (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['blue'])
            cv2.putText(frameColor, 'to track is the only visible blob.',
                (10,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['blue'])
            cv2.putText(frameColor, 'Press Space to set a new threshold value, Esc to exit.',
                (10,h-30), cv2.FONT_HERSHEY_DUPLEX, 0.6, BGR_COLOR['red'])

            frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))###
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
            _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if contours:
                contour = contours[np.argmax(map(cv2.contourArea, contours))]
                M = cv2.moments(contour)
                if M['m00']==0:
                    continue
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                # Draw the most acute angles of the contour (tail/muzzle/paws of the animal)
                hull = cv2.convexHull(contour)
                imgPoints = np.zeros_like(frame)
                for i in range(2, len(hull)-2):
                    if np.dot(hull[i][0]- hull[i-2][0], hull[i][0]- hull[i+2][0]) > 0:
                        imgPoints = cv2.circle(imgPoints, (hull[i][0][0],hull[i][0][1]), 5, BGR_COLOR['yellow'], -1, cv2.LINE_AA)

                # Draw a contour and a centroid of the animal
                cv2.drawContours(imgPoints, [contour], 0, BGR_COLOR['green'], 2, cv2.LINE_AA)
                imgPoints = cv2.circle(imgPoints, (x,y), 5, BGR_COLOR['black'], -1)

                imgContour = imgPoints
                frame = cv2.bitwise_and(frame, frame, mask = thresh)
                frame = cv2.addWeighted(frame, 0.4, imgContour, 1.0, 0.)
                cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)

            else:
                frame = np.zeros_like(frame)

            frame = cv2.resize(frame,(h*DimX/DimY,h))
            layout = np.hstack((frame, frameColor))
            cv2.imshow('Calibration', layout)

        cv2.destroyAllWindows()
        print("Calibration complete. Threshold set to %s." % AvF_thresh)


if __name__ == '__main__':
    from OFTrack import floorCrop, BGR_COLOR, counterclockwiseSort, load_mask
    #Argparsing
    parser = argparse.ArgumentParser(description='Animal tracking with OpenCV')
    parser.add_argument('-l','--live',dest='live',metavar='SRC',default='',
        help='Specify a camera for live video calibration. It can be an integer or an ip address.')
    parser.add_argument('-m','--mask',dest='mask',metavar='IMG',default='',help='Specify a mask image.')
    args = parser.parse_args()

    #Load data
    conf_data = reload()
    AvF_thresh = conf_data[-1]

    #Get mask's full path and load it
    if args.mask:
        args.mask = os.path.abspath(os.path.expanduser(args.mask))
        mask = load_mask(args.mask, conf_data)
        if mask is not None:
            print('Mask loaded correctly.')
        else: 
            print("Couldn't load mask correctly.")
        
    #Launch GUI
    getparams().mainloop()
