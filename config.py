import tkFileDialog as filedialog
import Tkinter as tk
import numpy as np
import os, cv2

#[Dimx,Dimy,CC,ratio,FPS,out_res,AvF_thresh]
DEFAULTS = np.array([33,21,0,3,30,2,170])

#Resolution options
#RES = ["1920x1080","1280x720","960x540", "640x360"]
RES = ["3840x1080","2560x720","1920x540","1280x360"]
#Color Config
CC = ["Dark Animal / Clear Surface", "Clear Animal / Dark Surface"] 
#Scaling options
SC = ["1/1","1/2","2/3","1/3","1/4"]


def reload():
    global DEFAULTS
    try:
        conf_data = np.genfromtxt("OFT.conf",delimiter=",",dtype=None)
        print("Config file loaded successfully.")
    except:
        print("Couldn't load config file properly.")
        np.savetxt('OFT.conf',DEFAULTS.reshape(1,7),delimiter=',',fmt='%s')
        conf_data = DEFAULTS
        print("New config file created.")
    return conf_data


class getparams(tk.Tk):

    def __init__(self):
        
        tk.Tk.__init__(self)
        self.wm_title("OF Tracker Configuration")
        self.eval('tk::PlaceWindow %s center' % self.winfo_pathname(self.winfo_id()))
        self.RES = tk.StringVar()
        self.CC = tk.StringVar()
        self.SC = tk.StringVar()

        #Labels
        self.ST1 = tk.Label(self,text='Experimental Settings',font=(20))
        self.ST2 = tk.Label(self,text='Online Visualization',font=(20))
        self.ST3 = tk.Label(self,text='Output',font=(20))
        self.Ldimx = tk.Label(self,text='Length(cm)')
        self.Ldimy = tk.Label(self,text='Width(cm)')
        self.Lcc = tk.Label(self,text='Color Configuration')
        self.Lsc = tk.Label(self,text='Scaling')
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
        self.save.grid(row=10, column=0, pady=10)
        self.exit.grid(row=10, column=1, pady=10)
        self.lift()

    #conf_data:[Dimx,Dimy,CC,sc,FPS,res,AvF_thresh]    
    def on_save(self):
        global DEFAULTS, RES, CC, SC, AvF_thresh, conf_data
        try:
            dimx = int(self.dimx.get())
            dimy = int(self.dimy.get())
            cc = [i for i in range(len(CC)) if CC[i]==self.CC.get()][0]
            sc = [i for i in range(len(SC)) if SC[i]==self.SC.get()][0]
            fps = int(self.fps.get())
            res = [i for i in range(len(RES)) if RES[i]==self.RES.get()][0]
            Settings = np.array([dimx,dimy,cc,sc,fps,res,AvF_thresh]).reshape(1,7)
            np.savetxt('OFT.conf',Settings,delimiter=',',fmt='%s')
            conf_data = np.array([dimx,dimy,cc,sc,fps,res,AvF_thresh])
            print(conf_data)
        except:
            print('saving failed.')
        
        
    def on_exit(self):
        self.destroy()
        exit()

    def on_cal(self):
        global conf_data,SC,AvF_thresh
        [DimX,DimY,cc,RA,FPS,res,THRESHOLD_ANIMAL_VS_FLOOR] = conf_data
        #res = RES[res].split('x')
        RA = SC[RA]
        RA = RA.split('/')
        ratio = float(RA[0])/float(RA[1])

        tk.Tk().withdraw()
        filename=filedialog.askopenfilename()
        if not filename:  return
        perspectiveMatrix = floorCrop(filename, conf_data)
        name = os.path.splitext(filename)[0]
        cap = cv2.VideoCapture(filename)

        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(h*ratio)
        w = int(w*ratio)
        
        ret, frame = cap.read()
        nullframes = 0
        while not frame.any():
            ret, frame = cap.read()
            print('no frames yet')
            nullframes += 1
            if nullframes > 50:
                break
        if ret: print('Frame found.')
        else: 
            print('No frames found.')
            return

        if  len(perspectiveMatrix[os.path.splitext(filename)[0]])==0:
            print('no perspective')
            return

        kernelSize = (25, 25)

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

            if k == 32:
                AvF_thresh = THRESHOLD_ANIMAL_VS_FLOOR
                self.thresh['text'] = 'Threshold: ' + str(AvF_thresh)
                
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            calframe = cv2.resize(frame,(w,h))
            frameColor = calframe.copy()

            #Text Shadows
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

            if not cc:
                calframe = cv2.bitwise_not(calframe)

            frame = cv2.warpPerspective(calframe, perspectiveMatrix[name], (w,h))###
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
            _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #print(THRESHOLD_ANIMAL_VS_FLOOR,len(contours))
            if len(contours)>0:
                contour = contours[np.argmax(map(cv2.contourArea, contours))]
                M = cv2.moments(contour)
                if M['m00']==0:
                    continue
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                # Draw the most acute angles of the contour (tail/muzzle/paws of the animal)
                hull = cv2.convexHull(contour)
                imgPoints = np.zeros(frame.shape,np.uint8)
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

            frame = cv2.resize(frame,(h*DimX/DimY,h))
            layout = np.hstack((frame, frameColor))
            cv2.imshow('Calibration', layout)

        cv2.destroyAllWindows()
        print("Calibration complete.")


if __name__ == '__main__':
    from OFTrack import floorCrop, BGR_COLOR
    conf_data = reload()
    AvF_thresh = conf_data[-1]
    getparams().mainloop()
