#!/usr/bin/python2.7
import numpy as np
import cv2
import os, sys, time, datetime
import Tkinter as tk
import tkFileDialog as filedialog
from os import chdir

#TODO Save positions as absolute coordinates based on analizyng box dimensions

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

def floorCrop(filename, conf_data):
    global perspectiveMatrix,tetragons,name,croppingPolygons,SD
    global RENEW_TETRAGON, ratio, DimX, DimY, CC, FPS, THRESHOLD_ANIMAL_VS_FLOOR
    ###########
    [DimX,DimY,CC,RA,FPS,res,THRESHOLD_ANIMAL_VS_FLOOR] = conf_data
    res = RES[res].split('x')
    SD = int(res[0]), int(res[1])
    RA = SC[RA]
    RA = RA.split('/')
    ratio = float(RA[0])/float(RA[1])
    ##############

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
        return perspectiveMatrix

def trace(filename):
    global perspectiveMatrix,croppingPolygons,tetragons,name,WAIT_DELAY
    global POS, DimX, DimY, SD, CC

    POS=np.array([[-1,-1,-1]])###
    name = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(filename)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(h*ratio)
    w = int(w*ratio)

    # Take first non-null frame and find corners within it
    ret, frame = cap.read()
    while not frame.any():
        ret, frame = cap.read()

    frame = cv2.resize(frame,(w,h))#############
    if not CC:
        frame = cv2.bitwise_not(frame)##########
    
    video = cv2.VideoWriter(RELATIVE_DESTINATION_PATH + 'timing/' + name + "_trace.avi", cv2.VideoWriter_fourcc(*'X264'), FPS, SD, cv2.INTER_LINEAR)
    #imgTrack = np.zeros_like(frame)
    imgTrack = np.zeros([ h, int(float(h)*float(DimX)/float(DimY)), 3 ],dtype='uint8')
    
    start = time.time()
    distance = _x = _y = 0
    Distance = x = y = 0
    
    while frame is not None:
        ret, frame = cap.read()
        
        if frame is None:   # not logical
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        
        frame = cv2.resize(frame,(w,h))
        frameColor = frame.copy()
        if not CC:
            frame = cv2.bitwise_not(frame)

        
        if len(croppingPolygons[name]) == 4:
            cv2.drawContours(frameColor, [np.reshape(croppingPolygons[name], (4,2))], -1, BGR_COLOR['red'], 2, cv2.LINE_AA)
        else:
            cv2.drawContours(frameColor, tetragons, -1, BGR_COLOR['red'], 2, cv2.LINE_AA)

        frame = cv2.warpPerspective(frame, perspectiveMatrix[name], (w,h))#####
        #frame = cv2.warpPerspective(frame, perspectiveMatrix[name], ( int(float(h)*float(DimX)/float(DimY) ), h))#####
        frame = cv2.resize(frame,( int(float(h)*float(DimX)/float(DimY) ), h))#############

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernelSize = (25, 25)
        frameBlur = cv2.GaussianBlur(frameGray, kernelSize, 0)
        _, thresh = cv2.threshold(frameBlur, THRESHOLD_ANIMAL_VS_FLOOR, 255, cv2.THRESH_BINARY)
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

        if ONLINE:
            if not contours:
                
                frame = cv2.add(np.zeros_like(frame), imgTrack)
                cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                    (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                    (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
                cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)
                
                layout = np.hstack((frame, frameColor))
                video.write(cv2.resize(layout, SD))
                cv2.imshow('Open Field Trace of ' + name, layout)
            
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

            cv2.putText(frame, "Distance " + str('%.2f' % Distance) + 'm',
                (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            cv2.putText(frame, "Time " + str('%.0f sec' % (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)),
                (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, BGR_COLOR['white'])
            cv2.circle(frame, (x,y), 5, BGR_COLOR['black'], -1, cv2.LINE_AA)
            
            layout = np.hstack((frame, frameColor))

            cv2.imshow('Open Field Trace of ' + name, layout)
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
        POS = np.append(POS,[[t,abs_x,abs_y]],axis=0)## ######################################
    
    POS = np.delete(POS,0,axis=0)###
    np.savetxt(RELATIVE_DESTINATION_PATH + 'positions/' + '[' + str(DimX) + 'x' + str(DimY) + '] ' + name + '.csv',
        POS, fmt = '%.2f', delimiter = ',')
    
    cv2.destroyAllWindows()
    cap.release()

    if ONLINE:
        video.release()
        cv2.imwrite(RELATIVE_DESTINATION_PATH + 'traces/' + name + '_[distance]=%.2f' % Distance + 'm' +
            '_[time]=%.1fs' % t + '.png', cv2.resize(imgTrack, (w, h))) ############
    
    print(filename + "\tdistance %.2f\t" % Distance + 'm ' + "processing/real time %.1f" % float(time.time()-start) + "/%.1f s" % t)
    file.write(name + ",%.2f" % Distance + ",%.1f\n" % t)
    file.close()



ONLINE = True

if len(sys.argv)>1 and '--offline' in sys.argv:
    ONLINE = False


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


#LOAD CONFIG
from config import RES, CC, SC, reload
if __name__ == '__main__':
    conf_data = reload()


if __name__ == '__main__':
    tk.Tk().withdraw()
    file_paths=filedialog.askopenfilenames()
    files = [file.split('/')[-1] for file in file_paths]
    paths =['/'.join(p)+'/' for p in [path.split('/')[:-1] for path in file_paths]]
    RELATIVE_DESTINATION_PATH = str(datetime.date.today()) + "_distance/"
    chdir(paths[0])
 
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'traces'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'traces')
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'positions'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'positions')

    file = open(RELATIVE_DESTINATION_PATH + "distances.csv", 'w')
    file.write("animal,distance [metres],run time [seconds]\n")
    file.close()

    for filename in files:
        file = open(RELATIVE_DESTINATION_PATH + "distances.csv", 'a')
        floorCrop(filename, conf_data)

