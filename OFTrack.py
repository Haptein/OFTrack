#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
from config import CC, SC, EXT, reload
import tkFileDialog as filedialog
import os, sys, time, datetime
import Tkinter as tk
import numpy as np
import argparse
import cv2

def counterclockwiseSort(tetragon):
    tetragon = sorted(tetragon, key = lambda e: e[0])
    tetragon[0:2] = sorted(tetragon[0:2], key = lambda e: e[1])
    tetragon[2:4] = sorted(tetragon[2:4], key = lambda e: e[1], reverse = True)
    return tetragon

def load_mask(mask_file,conf_data):
    global mask_cont, mask_croppingPolygons, mask_perspectiveMatrix
    try:
        #Read mask image and binarize it
        mask = cv2.imread(mask_file,0)
        mask_h, mask_w = mask.shape
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        #Get display ratio
        RA = SC[conf_data[3]].split('/')
        ratio = float(RA[0])/float(RA[1])
        #Resize to display size
        mask_w = int(mask_w*ratio)
        mask_h = int(mask_h*ratio)
        mask = cv2.resize(mask,(mask_w,mask_h))

        #Get contours, the minimum area rectangle containing the biggest contour and get its vertices
        ret,mask_cont,hier = cv2.findContours(mask, 1, 2)
        mask_cont = mask_cont[np.argmax(map(cv2.contourArea, mask_cont))]
        rect = cv2.minAreaRect(mask_cont)
        box = cv2.boxPoints(rect)

        #Generate perspective matrix
        mask_croppingPolygons = np.uint64(counterclockwiseSort(box))
        tetragonVertices = np.float32(mask_croppingPolygons)
        tetragonVerticesUpd = np.float32([[0,0],[0,mask_h],[mask_w,mask_h],[mask_w,0]])
        mask_perspectiveMatrix = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)

        #Make mask the same dimensions as frames read
        mask = np.dstack((mask,mask,mask))
        return mask
    except:
        return None

def progressBar(iteration, total, length = 50, fill = '█'):
    global time_params
    fiee = 120 #Frame interval for eta estimation

    #Each fiee frames time_params' lower gets updated
    if iteration%(2*fiee) == 0:
        time_params[1] = time.time()
    elif iteration%fiee == 0:
        time_params[0] = time.time()
    
    #ETA calculation
    eta = (total-iteration) / ( fiee / (max(time_params)-min(time_params)) )
    hors , secs = divmod(int(eta),3600)
    mins , secs = divmod(secs,60)

    #String formating
    prefix = 'File %s/%s'%(file_num+1,len(files))
    suffix = 'Time left: ' + ('%s hours, '%hors if hors>0 else '')\
        + ('%s minutes and '%mins if mins>0 else '') + '%s seconds.'%secs + 25*' '
    percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    #Print
    sys.stdout.write('%s: |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total and file_num+1 == len(files): 
        print('Completed!')


# Mouse callback function for drawing a cropping polygon
# This function is ignored if a mask is selected
def drawFloorCrop(event,x,y,flags,params):
    global perspectiveMatrix, name, RENEW_TETRAGON,END_SELECTION
    imgCroppingPolygon = np.zeros_like(params['imgFloorCorners'])

    #If key pressed r or R, reset selection
    if RENEW_TETRAGON:
            params['croppingPolygons'][name] = np.array([[0,0]])
            RENEW_TETRAGON = False
            cv2.imshow('Floor Corners for ' + name, params['imgFloorCorners'])
    
    #Last point selected
    if len(params['croppingPolygons'][name]) > 4 and event == cv2.EVENT_LBUTTONUP:
        ### Could remove this using global w,h instead.
        w = params['imgFloorCorners'].shape[1]
        h = params['imgFloorCorners'].shape[0]

        # delete 5th extra vertex of the floor cropping tetragon
        params['croppingPolygons'][name] = np.delete(params['croppingPolygons'][name], -1, 0)

        # Sort cropping tetragon vertices counter-clockwise starting with top left
        params['croppingPolygons'][name] = counterclockwiseSort(params['croppingPolygons'][name])

        # Get the matrix of perspective transformation
        params['croppingPolygons'][name] = np.reshape(params['croppingPolygons'][name], (4,2))
        tetragonVertices = np.float32(params['croppingPolygons'][name])
        cv2.destroyWindow('Floor Corners for ' + name)
        tetragonVerticesUpd = np.float32([[0,0],[0,h],[w,h],[w,0]])
        perspectiveMatrix[name] = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
        END_SELECTION = True

    #With every point selected
    if event == cv2.EVENT_LBUTTONDOWN:
        #First point selected
        if len(params['croppingPolygons'][name]) == 1:
            params['croppingPolygons'][name][0] = [x, y]
        #Add point to array
        params['croppingPolygons'][name] = np.append(params['croppingPolygons'][name], [[x, y]], axis=0)
    

    #If mouse moves and there's still less than 4 selected points redraw polygon area
    if event == cv2.EVENT_MOUSEMOVE and not (len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON):
        params['croppingPolygons'][name][-1] = [x, y]   
        if len(params['croppingPolygons'][name]) > 1:
            cv2.fillPoly(imgCroppingPolygon, [np.reshape(params['croppingPolygons'][name], (len(params['croppingPolygons'][name]),2))], BGR_COLOR['green'], cv2.LINE_AA)
            imgCroppingPolygon = cv2.addWeighted(params['imgFloorCorners'], 1.0, imgCroppingPolygon, 0.5, 0.)
            cv2.imshow('Floor Corners for ' + name, imgCroppingPolygon)

#This function gets the the stage area with user either user input or the help of masks
#and generates each file's perspective matrices for perspective correction.
#Its also used in config.py before the calibration process.
def floorCrop(filename, conf_data, args):
    global perspectiveMatrix,croppingPolygons,SD, name, mask_cont, mask_croppingPolygons, END_SELECTION
    global RENEW_TETRAGON, ratio, DimX, DimY, CC, FPS, THRESHOLD_ANIMAL_VS_FLOOR, cap, ext, mask, mask_perspectiveMatrix

    ########### Load config data # This vars are also used in the trace function
    [DimX,DimY,CC,RA,FPS,res,ext,THRESHOLD_ANIMAL_VS_FLOOR] = conf_data
    ext = EXT[ext]
    
    #Get Resolution
    SD = RESOLUTION[res]
    
    #Get ratio
    RA = SC[RA]
    RA = RA.split('/')
    ratio = float(RA[0])/float(RA[1])
    ###########
    
    #name is just an identifier for the file in a couple dicts
    if args.live:
            name = 'Live'
    else:        
            name = os.path.splitext(filename)[0]

    #Init vars
    tetragonVertices = []
    perspectiveMatrix[name] = []
    croppingPolygons[name] = np.array([[0,0]])

    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(h*ratio)
    w = int(w*ratio)

    #Load mask if run from config.py
    if __name__ != '__main__':
         mask = load_mask(args.mask, conf_data)

    #If mask enabled, mask loaded correctly, and sizes are the same
    if (mask is not None) and mask.shape == (h,w,3):
        #Reuse the mask perspective matrix and croppingpols
        croppingPolygons[name] = mask_croppingPolygons
        perspectiveMatrix[name] = mask_perspectiveMatrix    
        #File ready to track|calibrate
        if __name__ == '__main__':
            cap.release()
            return
        else:
            cap.release()
            return perspectiveMatrix
    elif args.mask:
        print('The size of %s does not match with that of the mask provided. Please select a region manually.'%filename)

    #Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while frame is None:
        ret, frame = cap.read()
        print('no frames yet')

    frame = cv2.resize(frame,(w,h))
    #Turn grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #and back to BGR so we can overlay the selected polygon in color
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
    imgFloorCorners = frameGray

    END_SELECTION = False
    cv2.imshow('Floor Corners for ' + name, imgFloorCorners)
    cv2.setMouseCallback('Floor Corners for ' + name, drawFloorCrop, {'imgFloorCorners': imgFloorCorners, 'croppingPolygons': croppingPolygons})

    while not END_SELECTION:    
        #Read key presses
        k = cv2.waitKey(0)

        #Esc to exit
        if k == 27:
            if __name__ == '__main__':
                sys.exit()
            else:
                cv2.destroyWindow('Floor Corners for ' + name)    
                cap.release()
                END_SELECTION = True

        #Press r or R to reset selection
        if k == 114 or k == 82:
            RENEW_TETRAGON = True

        #If window is closed by any other means
        if not END_SELECTION:
            try:
                _ = cv2.getWindowProperty('Floor Corners for ' + name,0)
            except:
                if __name__ == '__main__':
                    sys.exit()
                else:
                    cv2.destroyWindow('Floor Corners for ' + name)    
                    cap.release()
                    END_SELECTION = True
        
    cv2.destroyWindow('Floor Corners for ' + name)
    if __name__ == '__main__':
        cap.release()
        return
    else:
        cap.release()
        return perspectiveMatrix

#This is where the tracking is done.
def trace(filename):
    global perspectiveMatrix,croppingPolygons,WAIT_DELAY
    global DimX, ratio, DimY, SD, CC, ext, mask, mask_cont

    POS=np.array([[-1,-1,-1]])#Init file for the csv dump
    kernelSize = (25, 25)#Kernel size for GaussianBlurring (normal thresholding)

    #File id
    if args.live:
        name = 'Live'
        livedate = time.strftime(" %Y-%m-%d[%H:%M:%S]")
    else:
        name = os.path.splitext(filename)[0]
        livedate = ''

    #Init VideoCapture, and get video dimensions
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(h*ratio)
    w = int(w*ratio)

    #If trace overlay is enabled mantain video aspect ratio and generate the inverse perspective matrix
    if args.overlay:
        #SD isn't modified in floorcrop bc its used by config.py
        aspect_ratio = float(w)/float(h)
        SD = int(SD[1]*aspect_ratio) , SD[1]
        re, invper = cv2.invert(perspectiveMatrix[name])

    #Perhaps we dont need to re-read a frame if we've got the one read in filecrop
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()

    #If automatic background subtraction is enabled
    if args.abs:
        #Create background subtractor object
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,history=600,varThreshold=72)

        #Feed the first frame to it
        frame = cv2.resize(frame,(w,h))
        if not CC:
            frame = cv2.bitwise_not(frame)
        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))
        frame = cv2.resize(frame,( int(float(h)*float(DimX)/float(DimY) ), h))
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
        thresh = fgbg.apply(frameBlur)
    
    #If video output enabled
    if args.out_video:
            video = cv2.VideoWriter(RELATIVE_DESTINATION_PATH + 'timing/' + name + livedate + "_trace." + ext,
                cv2.VideoWriter_fourcc(*'X264'), FPS, SD, cv2.INTER_LINEAR)

    #Init array containing trace
    imgTrack = np.zeros([ h, int(float(h)*float(DimX)/float(DimY)), 3 ],dtype='uint8')
    
    #Init distance variables
    #distance - Relative (in terms of selected area height)
    #Distance - Absolute (derived from values entered in config)
    distance = _x = _y = 0
    Distance = x = y = 0
    
    first_contour = True
    #Read frames until the end of time, or frames.
    while frame is not None:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Get time
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        
        #Resize frame
        frame = cv2.resize(frame,(w,h))
        frameColor = frame.copy()

        #CC is true for "Clear Animal / Dark Surface"
        #and false for "Dark Animal / Clear Surface"
        if not CC:
            frame = cv2.bitwise_not(frame)
        
        if args.mask and mask.shape == (h,w,3):
            #frameColor = frameColor * mask
            frame = frame * mask
        
        #Draw selected area
        if args.mask and mask.shape == (h,w,3):
            #Mask Contour
            cv2.drawContours(frameColor, [mask_cont], -1, BGR_COLOR['black'], 2, cv2.LINE_AA)
        else:
            #Selected polygon
            cv2.drawContours(frameColor, [np.reshape(croppingPolygons[name], (4,2))], -1, BGR_COLOR['black'], 2, cv2.LINE_AA)
        

        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))#####
        frame = cv2.resize(frame,( int(float(h)*float(DimX)/float(DimY) ), h))#####
        #Find a way to do this ^^^ in one step
        #frame = cv2.warpPerspective(frame, perspectiveMatrix[name], ( int(float(h)*float(DimX)/float(DimY) ), h))#####

        if not args.abs:
            #Normal thresholding
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
            _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
        else:
            #Background Subtraction
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameBlur = cv2.GaussianBlur(frameGray, (15,15), 0)
            thresh = fgbg.apply(frameBlur)

        #Get contours
        _, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            #If abs disabled or beginning of video
            if not args.abs or t<3 or first_contour:
                # Find a contour with the biggest area (the animal if you set your stuff correctly)
                contour = contours[np.argmax(map(cv2.contourArea, contours))]
                M = cv2.moments(contour)
                if M['m00']==0: continue
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                if _x == 0 and _y == 0:
                    _x = x
                    _y = y

                if args.abs:
                    _a = np.argmax(map(cv2.contourArea, contours))
                    
            else:
                #Get area and positions for every contour. Animal is point closer to last in coordinates(Area,distance_from_last_point)
                #Could also try matching shapes.
                criteria_space = []#Array containing [area,distance_from_last_point] for each contour
                areas_cont = map(cv2.contourArea, contours)
                momes_cont = map(cv2.moments, contours)
                for a,m in zip(areas_cont,momes_cont):
                    if m['m00']==0:
                        criteria_space.append(np.inf)
                        continue
                    #Get center of contour and calculate distance to last point
                    xx = int(m['m10']/m['m00'])
                    yy = int(m['m01']/m['m00'])
                    dis = ( np.sqrt( (xx-_x)**2 + (yy-_y)**2 ))
                    if dis/float(h) > 0.2  : #If distance is from last point ist greater than 20% of frame's height
                        criteria_space.append(np.inf)    #It's most probably not the object to track
                        continue

                    criteria_space.append( np.sqrt( (a-_a)**2 + (dis-0)**2 ) )
                    
                #Contour must be the one with minimum distance in this space.
                contour = contours[np.argmin(criteria_space)]
                M = momes_cont[np.argmin(criteria_space)]
                if M['m00']==0: continue
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                _a = cv2.contourArea(contour)

            first_contour = False
            distance += np.sqrt( (x-_x)**2 + (y-_y)**2 )/float(h)
            Distance = distance*DimY/100
        else:   
            #Update cli progress bar
            if not args.live:
                progressBar(cap.get(cv2.CAP_PROP_POS_FRAMES),cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
        if args.display or args.out_video:
            if not contours:
                
                frame = imgTrack.copy()
                
                if args.overlay:
                    #Inverse perspective transformation
                    invimgTrack = cv2.warpPerspective(cv2.resize(frame,(w,h)), invper, (w,h), cv2.WARP_INVERSE_MAP)
                    frame = cv2.addWeighted(frameColor, 0.8,invimgTrack, 0.4, 0)

                if args.video_dist:
                    cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                        (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                if args.video_time:
                    cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                        (20,20*(1 + args.video_dist)), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            
                if args.overlay:
                    layout = frame
                else:
                    layout = np.hstack((frame, frameColor))
            
                if args.display:
                    cv2.imshow('Open Field Trace of ' + name, layout)

                if args.out_video:
                    video.write(cv2.resize(layout, SD))
            
                k = cv2.waitKey(WAIT_DELAY) & 0xff
                if k == 27:
                    print('\nTracking %s interrupted.'%filename)
                    break
                if k == 32:
                    if WAIT_DELAY == 1:
                        WAIT_DELAY = 0  # pause
                    else:
                        WAIT_DELAY = 1  # play as fast as possible
                continue

            # Draw the most acute angles of the contour (tail/muzzle/paws of the animal)
            hull = cv2.convexHull(contour)
            imgPoints = np.zeros(frame.shape,np.uint8)
            for i in range(2, len(hull)-2):
                if np.dot(hull[i][0]- hull[i-2][0], hull[i][0]- hull[i+2][0]) > 0:
                    imgPoints = cv2.circle(imgPoints, (hull[i][0][0],hull[i][0][1]), 5, BGR_COLOR['yellow'], -1, cv2.LINE_AA)

            # Draw contour and centroid of the animal
            cv2.drawContours(imgPoints, [contour], 0, BGR_COLOR['green'], 2, cv2.LINE_AA)
            imgPoints = cv2.circle(imgPoints, (x,y), 5, BGR_COLOR['black'], -1)

            # Draw track of the animal
            if args.live: #CAP_PROP_POS_AVI_RATIO isn't supported for cameras
                imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 1, cv2.line(imgTrack, (x,y), (_x,_y),
                    (255, 127, 255), 1, cv2.LINE_AA), 0.99, 0.)
            else:
                imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 1, cv2.line(imgTrack, (x,y), (_x,_y),
                    (255, 127, int(cap.get(cv2.CAP_PROP_POS_AVI_RATIO)*255)), 1, cv2.LINE_AA), 0.99, 0.)

            imgContour = cv2.add(imgPoints, imgTrack)

            frame = cv2.bitwise_and(frame, frame, mask = thresh)
            frame = cv2.addWeighted(frame, 0.4, imgContour, 1.0, 0.)
            cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)

            if args.overlay:
                invimgTrack = cv2.warpPerspective(cv2.resize(frame,(w,h)), invper, (w,h), cv2.WARP_INVERSE_MAP)
                frame = cv2.addWeighted(frameColor, 0.8,invimgTrack, 0.4, 0)

            if args.video_dist:
                cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                    (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            if args.video_time:
                cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                    (20,20*(1 + args.video_dist)), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                        
            if args.overlay:
                layout = frame
            else:
                layout = np.hstack((frame, frameColor))
            
            if args.display:
                cv2.imshow('Open Field Trace of ' + name, layout)

            if args.out_video:
                video.write(cv2.resize(layout, SD))            

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if k == 27:
                print('\nTracking %s interrupted.'%filename)
                break
            if k == 32:
                if WAIT_DELAY == 1:
                    WAIT_DELAY = 0  # pause
                else:
                    WAIT_DELAY = 1  # play as fast as possible
        _x = x
        _y = y
        abs_x = float(DimY)*float(x)/float(h)
        abs_y = float(y)/float(h)*float(DimY)

        if args.out_csv:
            POS = np.append(POS,[[t,abs_x,abs_y]],axis=0)# Time & XY Positions for csv file

        #Update cli progress bar
        if not args.live:
            progressBar(cap.get(cv2.CAP_PROP_POS_FRAMES),cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.out_csv:
        POS = np.delete(POS,0,axis=0)
        np.savetxt(RELATIVE_DESTINATION_PATH + 'positions/' + '[' + str(DimX) + 'x' + str(DimY) + '] ' + name + '.csv',
            POS, fmt = '%.2f', delimiter = ',')
    
    cv2.destroyAllWindows()
    cap.release()

    if args.out_video:
        video.release()
    


#Init some vars
BGR_COLOR = {'red': (0,0,255),
        'green': (127,255,0),
        'blue': (255,127,0),
        'yellow': (0,127,255),
        'black': (0,0,0),
        'white': (255,255,255)}
WAIT_DELAY = 1
RENEW_TETRAGON = False
perspectiveMatrix = dict()
croppingPolygons = dict()
time_params = [time.time(),time.time()+1]
RESOLUTION = [(3840,1080),(2560,720),(1920,540),(1280,360)]


if __name__ == '__main__':
    #Load config
    conf_data = reload()
    #Argparsing
    parser = argparse.ArgumentParser(description='Animal tracking with OpenCV')
    parser.add_argument('input',nargs='*',help='Input files.')
    parser.add_argument('-o','--output',dest='out_destination',metavar='DES',default='',help='Specify output destination.')
    parser.add_argument('-m','--mask',dest='mask',metavar='IMG',default='',help='Specify a mask image.')
    parser.add_argument('-a','--abs',dest='abs',action='store_true',help="Enable automatic background subtraction based tracking.")
    parser.add_argument('-ov','--overlay',dest='overlay',action='store_true',help='Overlay video with trace instead of side by side view.')
    parser.add_argument('-nv','--no-video',dest='out_video',action='store_false',help='Disable video file output.')
    parser.add_argument('-nc','--no-csv',dest='out_csv',action='store_false',help='Disable csv file output.')
    parser.add_argument('-nd','--no-display',dest='display',action='store_false',help='Disable video display.')
    parser.add_argument('-l','--live',dest='live',metavar='SRC',default='',
        help='Specify a camera for live video feed. It can be an integer or an ip address.')
    parser.add_argument('-ht','--hide-time',dest='video_time',action='store_false',help="Hide time.")
    parser.add_argument('-hd','--hide-distance',dest='video_dist',action='store_false',help="Hide distance estimation.")
    args = parser.parse_args()

    #Get full paths
    file_paths = [os.path.abspath(os.path.expanduser(values)) for values in args.input]
    if args.out_destination:
        args.out_destination = os.path.abspath(os.path.expanduser(args.out_destination)) + '/'
    if args.mask:
        args.mask = os.path.abspath(os.path.expanduser(args.mask))
    
    #Load mask
    mask = load_mask(args.mask, conf_data)
    if mask is not None:
        print('Mask loaded correctly.')
    else:
        if args.mask:
            print("Couldn't load mask correctly.")
        
    #GUI file selection if no file or --live flag entered
    if args.live:
        if len(args.live)<3:
            live_camera = int(args.live)
        else:
            live_camera = args.live
        files = [live_camera]
    else:
        if not file_paths:
            tk.Tk().withdraw()
            file_paths=filedialog.askopenfilenames()
            if not file_paths:
                sys.exit()     
        files = [file.split('/')[-1] for file in file_paths]
        paths =['/'.join(p)+'/' for p in [path.split('/')[:-1] for path in file_paths]]
        os.chdir(paths[0])


    #Folder structure    
    RELATIVE_DESTINATION_PATH = args.out_destination + 'OFTrack [' + str(datetime.date.today()) + "]/"
    if args.out_video:
        if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
            os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
    if args.out_csv:
        if not os.path.exists(RELATIVE_DESTINATION_PATH + 'positions'):
            os.makedirs(RELATIVE_DESTINATION_PATH + 'positions')

    for filename in files:
        floorCrop(filename, conf_data, args)
    for file_num, filename in enumerate(files):
        trace(filename)

