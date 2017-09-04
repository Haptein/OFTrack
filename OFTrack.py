#!/usr/bin/python2.7
from config import RES, CC, SC, EXT, reload
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


# mouse callback function for drawing a cropping polygon
def drawFloorCrop(event,x,y,flags,params):
    global perspectiveMatrix,name,RENEW_TETRAGON
    imgCroppingPolygon = np.zeros_like(params['imgFloorCorners'])
    if event == cv2.EVENT_RBUTTONUP:
        cv2.destroyWindow('Floor Corners for ' + name)
    if len(params['croppingPolygons'][name]) > 4 and event == cv2.EVENT_LBUTTONUP:
        #RENEW_TETRAGON = True##################################################################################
        w = params['imgFloorCorners'].shape[1]###
        h = params['imgFloorCorners'].shape[0]
        params['croppingPolygons'][name] = np.delete(params['croppingPolygons'][name], -1, 0)   # delete 5th extra vertex of the floor cropping tetragon
        # Sort cropping tetragon vertices counter-clockwise starting with top left
        params['croppingPolygons'][name] = counterclockwiseSort(params['croppingPolygons'][name])
        # Get the matrix of perspective transformation
        params['croppingPolygons'][name] = np.reshape(params['croppingPolygons'][name], (4,2))
        tetragonVertices = np.float32(params['croppingPolygons'][name])
        cv2.destroyWindow('Floor Corners for ' + name)
        tetragonVerticesUpd = np.float32([[0,0],[0,h],[w,h],[w,0]])
        perspectiveMatrix[name] = cv2.getPerspectiveTransform(tetragonVertices, tetragonVerticesUpd)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON:
            params['croppingPolygons'][name] = np.array([[0,0]])
            RENEW_TETRAGON = False
        if len(params['croppingPolygons'][name]) == 1:
            params['croppingPolygons'][name][0] = [x, y]
        params['croppingPolygons'][name] = np.append(params['croppingPolygons'][name], [[x, y]], axis=0)
    if event == cv2.EVENT_MOUSEMOVE and not (len(params['croppingPolygons'][name]) == 4 and RENEW_TETRAGON):
        params['croppingPolygons'][name][-1] = [x, y]   
        if len(params['croppingPolygons'][name]) > 1:
            cv2.fillPoly(imgCroppingPolygon, [np.reshape(params['croppingPolygons'][name], (len(params['croppingPolygons'][name]),2))], BGR_COLOR['green'], cv2.LINE_AA)
            imgCroppingPolygon = cv2.addWeighted(params['imgFloorCorners'], 1.0, imgCroppingPolygon, 0.5, 0.)
            cv2.imshow('Floor Corners for ' + name, imgCroppingPolygon)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def floorCrop(filename, conf_data, args):
    global perspectiveMatrix,tetragons,croppingPolygons,SD, name
    global RENEW_TETRAGON, ratio, DimX, DimY, CC, FPS, THRESHOLD_ANIMAL_VS_FLOOR, cap, ext
    ########### Load config data
    [DimX,DimY,CC,RA,FPS,res,ext,THRESHOLD_ANIMAL_VS_FLOOR] = conf_data
    res = RES[res].split('x')
    ext = EXT[ext]
    SD = int(res[0]), int(res[1])
    RA = SC[RA]
    RA = RA.split('/')
    ratio = float(RA[0])/float(RA[1])
    ##############
    
    
    if args.live:
            name = 'Live'
    else:        
            name = os.path.splitext(filename)[0]
        
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(h*ratio)
    w = int(w*ratio)

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while frame is None:#not frame.any():
        ret, frame = cap.read()
        print('no frames yet')

    frame = cv2.resize(frame,(w,h))#####################

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tetragons = []

    perspectiveMatrix[name] = []
    croppingPolygons[name] = np.array([[0,0]])########
    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)#########
    tetragonVertices = []#########
    imgFloorCorners = frameGray###############
    cv2.imshow('Floor Corners for ' + name, imgFloorCorners)
    
    #cv2.namedWindow('Floor Corners for ' + name, cv2.WINDOW_NORMAL)#######3
    #cv2.resizeWindow('Floor Corners for ' + name, w,h)#########

    cv2.setMouseCallback('Floor Corners for ' + name, drawFloorCrop, {'imgFloorCorners': imgFloorCorners, 'croppingPolygons': croppingPolygons})
    k = cv2.waitKey(0)
    if k == 27:
        if __name__ == '__main__':
            sys.exit()
        else:
            cv2.destroyWindow('Floor Corners for ' + name)    
            cap.release()

    cv2.destroyWindow('Floor Corners for ' + name)
    if __name__ == '__main__':
        trace(filename)
    else:
        cap.release()
        return perspectiveMatrix

def trace(filename):
    global perspectiveMatrix,croppingPolygons,tetragons,name,WAIT_DELAY
    global POS, DimX, DimY, SD, CC, cap, ext, mask

    POS=np.array([[-1,-1,-1]])
    kernelSize = (25, 25)

    if args.abs:
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,history=500,varThreshold=64)

    if args.live:
        name = 'Live'
        livedate = time.strftime(" %Y-%m-%d[%H:%M:%S]")
    else:
        name = os.path.splitext(filename)[0]
        livedate = ''
    
    if args.mask:
        mask =  cv2.resize(mask,(w,h))
        mask = np.dstack((mask,mask,mask))

    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(h*ratio)
    w = int(w*ratio)

    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    #Process first frame if automatic bg subtraction is enabled
    if args.abs:
        frame = cv2.resize(frame,(w,h))
        if not CC:
            frame = cv2.bitwise_not(frame)
        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))
        frame = cv2.resize(frame,( int(float(h)*float(DimX)/float(DimY) ), h))
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
        thresh = fgbg.apply(frameBlur)
    
    if args.out_video:
            video = cv2.VideoWriter(RELATIVE_DESTINATION_PATH + 'timing/' + name + livedate + "_trace." + ext,
                cv2.VideoWriter_fourcc(*'X264'), FPS, SD, cv2.INTER_LINEAR)

    #Init array containing trace
    imgTrack = np.zeros([ h, int(float(h)*float(DimX)/float(DimY)), 3 ],dtype='uint8')
    
    start = time.time()
    distance = _x = _y = 0
    Distance = x = y = 0
    
    while frame is not None:
        ret, frame = cap.read()
        if not ret:
            break
        
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        
        frame = cv2.resize(frame,(w,h))
        frameColor = frame.copy()

        if not CC:
            frame = cv2.bitwise_not(frame)

        if args.mask:
            frameColor = frameColor * mask
            frame = frame * mask

        if len(croppingPolygons[name]) == 4:
            cv2.drawContours(frameColor, [np.reshape(croppingPolygons[name], (4,2))], -1, BGR_COLOR['red'], 2, cv2.LINE_AA)
        else:
            cv2.drawContours(frameColor, tetragons, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))#####
        #frame = cv2.warpPerspective(frame, perspectiveMatrix[name], ( int(float(h)*float(DimX)/float(DimY) ), h))#####
        frame = cv2.resize(frame,( int(float(h)*float(DimX)/float(DimY) ), h))#############

        if not args.abs:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
            _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
        else:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameBlur = cv2.GaussianBlur(frameGray, (15,15), 0)
            thresh = fgbg.apply(frameBlur)

        _, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            # Find a contour with the biggest area (animal most likely)
            contour = contours[np.argmax(map(cv2.contourArea, contours))]
            M = cv2.moments(contour)
            if M['m00']==0: continue
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            if _x == 0 and _y == 0:
                _x = x
                _y = y
        
            distance += np.sqrt( (x-_x)**2 + (y-_y)**2 )/float(h)
            Distance = distance*DimY/100

        
        if args.display or args.out_video:
            if not contours:
                
                frame = cv2.add(np.zeros_like(frame), imgTrack)
                if args.video_dist:
                    cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                        (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                if args.video_time:
                    cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                        (20,20*(1 + args.video_dist)), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)
                
                layout = np.hstack((frame, frameColor))

                if args.display:
                    cv2.imshow('Open Field Trace of ' + name, layout)

                if args.out_video:
                    video.write(cv2.resize(layout, SD))
            
                k = cv2.waitKey(WAIT_DELAY) & 0xff
                if k == 27:
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

            # Draw a contour and a centroid of the animal
            cv2.drawContours(imgPoints, [contour], 0, BGR_COLOR['green'], 2, cv2.LINE_AA)
            imgPoints = cv2.circle(imgPoints, (x,y), 5, BGR_COLOR['black'], -1)

            # Draw a track of the animal
            imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 0.85, cv2.line(imgTrack, (x,y), (_x,_y),
                (255, 127, int(cap.get(cv2.CAP_PROP_POS_AVI_RATIO)*255)), 1, cv2.LINE_AA), 0.98, 0.)

            imgContour = cv2.add(imgPoints, imgTrack)

            frame = cv2.bitwise_and(frame, frame, mask = thresh)
            frame = cv2.addWeighted(frame, 0.4, imgContour, 1.0, 0.)

            if args.video_dist:
                cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                    (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            if args.video_time:
                cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                    (20,20*(1 + args.video_dist)), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            
            cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)
            
            layout = np.hstack((frame, frameColor))

            if args.display:
                cv2.imshow('Open Field Trace of ' + name, layout)

            if args.out_video:
                video.write(cv2.resize(layout, SD))            

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if k == 27:
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
            POS = np.append(POS,[[t,abs_x,abs_y]],axis=0)# Time & Positions for csv file
    
    if args.out_csv:
        POS = np.delete(POS,0,axis=0)
        np.savetxt(RELATIVE_DESTINATION_PATH + 'positions/' + '[' + str(DimX) + 'x' + str(DimY) + '] ' + name + '.csv',
            POS, fmt = '%.2f', delimiter = ',')
    
    cv2.destroyAllWindows()
    cap.release()

    if args.out_video:
        video.release()
    
    print(filename + "\tdistance %.2f\t" % Distance + 'm ' + "processing/real time %.1f" % float(time.time()-start) + "/%.1f s" % t)
    file.write(name + ",%.2f" % Distance + ",%.1f\n" % t)
    file.close()


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
tetragons = []
name = ""



if __name__ == '__main__':
    #Load config
    conf_data = reload()
    #Argparsing
    parser = argparse.ArgumentParser(description='Animal tracking with OpenCV')
    parser.add_argument('input',nargs='*',help='Input files.')
    parser.add_argument('-o','--output',dest='out_destination',metavar='DES',default='',help='Specify output destination.')
    parser.add_argument('-m','--mask',dest='mask',metavar='IMG',default='',help='Specify a mask image.')
    parser.add_argument('-a','--abs',dest='abs',action='store_true',help="Automatic background subtraction.")
    parser.add_argument('-nv','--no-video',dest='out_video',action='store_false',help='Disable video file output.')
    parser.add_argument('-nc','--no-csv',dest='out_csv',action='store_false',help='Disable csv file output.')
    parser.add_argument('-nd','--no-display',dest='display',action='store_false',help='Disable video display.')
    parser.add_argument('-l','--live',dest='live',metavar='SRC',default='',
        help='Specify a camera for live video feed. It can be an integer or an ip address.')
    parser.add_argument('-ht','--hide-time',dest='video_time',action='store_false',help="Hide time.")
    parser.add_argument('-hd','--hide-distance',dest='video_dist',action='store_false',help="Hide distance estimation.")
    args = parser.parse_args()

    file_paths = [os.path.abspath(os.path.expanduser(values)) for values in args.input]
    if args.mask:
        mask = cv2.imread(args.mask,0)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    if args.live:
        files = [args.live]
    else:
        if not file_paths:
            tk.Tk().withdraw()
            file_paths=filedialog.askopenfilenames()

        files = [file.split('/')[-1] for file in file_paths]
        paths =['/'.join(p)+'/' for p in [path.split('/')[:-1] for path in file_paths]]
        os.chdir(paths[0])
    
    RELATIVE_DESTINATION_PATH = args.out_destination + 'OFTrack [' + str(datetime.date.today()) + "]/"
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'positions'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'positions')

    file = open(RELATIVE_DESTINATION_PATH + "distances.csv", 'w')
    file.write("animal,distance [metres],run time [seconds]\n")
    file.close()

    for filename in files:
        file = open(RELATIVE_DESTINATION_PATH + "distances.csv", 'a')
        floorCrop(filename, conf_data, args)

