import numpy as np
import cv2
import time
import cvzone
import math
import pandas as pd

# Optical Flow Pyramid
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

feature_params = dict(maxCorners = 1000,
                    qualityLevel = 0.3,
                    minDistance = 7,
                    blockSize = 7 )


trajectory_len = 10     
detect_interval = 1     
trajectories = []       
frame_idx = 0           
trajectoriesNum=0
energy_record = []      

cap = cv2.VideoCapture("Nayong.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')       
fps = cap.get(cv2.CAP_PROP_FPS)
size = (2*int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),2*int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
out = cv2.VideoWriter('Nayong4Stacked.avi',fourcc ,fps, size)   


while True:

    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()
       
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
    img = frame.copy()

 
    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        total_energy = 0  
        img0, img1 = prev_gray, frame_gray  
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)  
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)     
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)    
        d = abs(p0-p0r).reshape(-1, 2).max(-1)  
        good = d < 1    

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue        

            u = x - trajectory[-1][0]  
            v = y - trajectory[-1][1]  
            total_energy += u ** 2 + v ** 2     # Calculate optical flow energy

            trajectory.append((x, y))   
            if len(trajectory) > trajectory_len:    
                del trajectory[0]
            new_trajectories.append(trajectory)     
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)   

        trajectories = new_trajectories

        trajectoriesNum=0
        for trajectory in trajectories:
            pf = np.array(trajectory[0])       
            pl = np.array(trajectory[-1])
            p3 = pl-pf
            p4 = math.hypot(p3[0],p3[1])
            if p4>10:
                t = len(trajectory)/30
                speed = p4/t
                cv2.polylines(img, [np.int32(trajectory)], False, (0, 255, 0), thickness=2)
                trajectoriesNum += 1

        cv2.putText(img, 'Track Count: %d' % trajectoriesNum, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #### Record Optical Flow Energy
        df = pd.DataFrame({'Frame_Index': [frame_idx], 'OpticalFlow_Energy': [total_energy]})
        df.to_csv('optical_flow_energy.csv', mode='a', header=not bool(frame_idx - 1), index=False)
      

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)        
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:  
            cv2.circle(mask, (x, y), 5, 0, -1)      

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)  
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])   


    frame_idx += 1  
    print(frame_idx)
    # print(len(trajectories))
    #print(trajectoriesNum)
    prev_gray = frame_gray      

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    
    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    stackedImg = cvzone.stackImages([frame,frame_gray,mask,img], cols=2,scale=1)    
    out.write(stackedImg)       
    cv2.imshow('Mask', mask)
    cv2.imshow('Optical Flow',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


