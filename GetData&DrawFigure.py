import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#### Read all data between two frames
def extract_lines_between_frames(file_path, start_frame, end_frame):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_index = None
    end_index = None

    # Find the start and end indices
    for i, line in enumerate(lines):
        if line.strip() == f"Frame: {start_frame}":
            start_index = i
        elif line.strip() == f"Frame: {end_frame}":
            end_index = i
            break

    # Extract lines between the start and end frames
    if start_index is not None and end_index is not None and start_index < end_index:
        return lines[start_index + 1:end_index]
    else:
        return []


file_path = 'Alldata.txt'

for second in range(110, 111):

    start_frame = second*30      
    end_frame = start_frame+1   
    line_number = 1 

    extracted_lines = extract_lines_between_frames(file_path, start_frame, end_frame)


    speeds = []
    x_labels = []
    y_labels = []       

    for line in extracted_lines:
        if line_number >= 2:
            if line_number % 2 == 0:  
                match = re.search(r"[-+]?\d*\.\d+|\d+", line)   
                if match:
                    speed = float(match.group())
                    speeds.append(speed)

            else:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)        
                if matches and len(matches) >= 2:
                    
                    x_label = float(matches[-2])
                    y_label = float(matches[-1])
                    
                    x_labels.append(x_label)
                    y_labels.append(2160-y_label)
        line_number+=1

   

    fig = plt.figure()  
    sub = fig.add_subplot(111)

    scat = sub.scatter(x_labels,y_labels,marker='o',c=speeds,cmap='rainbow',s=5)  

    ax = plt.gca()
    ax.set_aspect(1)

    cb = fig.colorbar(scat,orientation='horizontal')  
    cb.ax.set_xlabel('Speed')       

    sub.set_xlim([0,width of original video])
    sub.set_ylim([0,height of original video])  

    plt.show()
    fig.savefig(f"Speed{second}s.png",transparent=True,dpi=300)
