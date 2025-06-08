import numpy as np
import cv2
import time
import cvzone
import matplotlib.pyplot as plt
import os
import pandas as pd

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)     
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def draw_h_channel(flow):      
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    h_channel = (ang * (180 / np.pi / 2)).astype(np.uint8)

    return cv2.cvtColor(h_channel, cv2.COLOR_GRAY2BGR), h_channel

def draw_v_channel(flow):       
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    v_channel = np.sqrt(fx * fx + fy * fy)

    v_channel = np.minimum(v_channel * 4, 255).astype(np.uint8)
    return cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR), v_channel


cap = cv2.VideoCapture("Nayong.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')        
fps = cap.get(cv2.CAP_PROP_FPS)
size = (2*int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),2*int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
out = cv2.VideoWriter('NayongDenseHSVSeperate.avi',fourcc ,fps, size)   

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

frame_number = 0  

while True:

    suc, img = cap.read()
    if not suc:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()


    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    prevgray = gray     


    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    print(f"{fps:.2f} FPS")

    draw_h, h_channel = draw_h_channel(flow)
    draw_v, v_channel = draw_v_channel(flow)
   
    stackedImg = cvzone.stackImages([draw_flow(gray, flow), draw_hsv(flow), draw_h, draw_v], cols=2, scale=1)  
    out.write(stackedImg)  
    cv2.imshow('flow', draw_flow(gray, flow))
    cv2.imshow('flow HSV', draw_hsv(flow))

    frame_number += 1  
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
