# Optical-Flow-Seismic-Rock-Avalanche-Detection
Optical-Flow-Seismic-Rock-Avalanche-Detection is a package to detect deformation and dynamic characteristics with computer vision and seismic signal analysis. The manuscript "A Novel Joint Framework of Computer Vision and Seismic Signal Analysis to Detect Deformation and Dynamic Characteristics of Large Rock Avalanches" has been submitted to Geophysical Research Letters. Once the manuscript is accepted and published, the paper will be uploaded in the repository and the repository will be updated.
# Recommended Interpreter
* Python 3.10 with packages opencv-python (4.9.0), matplotlib (3.9.0), numpy (1.26.4), pandas (2.2.2), cvzone (1.6.1), sympy (1.12) installed
* Windows 10/11
# Description of Files
* **denseOpticalFlow.py** and **sparseOpticalFlow.py** contain the code necessary to implement the improved Optical Flow methods.
* **GetData&DrawFigure.py** is used for post-processing data derived from the Lucas-Kanade sparse optical flow method.
* **HVHistograms.py** is used to draw H&V channel histograms based on the results derived from the Farnebäck dense optical flow method.
# Instructions
1. Download the four python code files (.py) and put them into an overarching folder somewhere in your file system.
2. Build a Python programing environment and ensure that required Python packages are installed.
3. Change the video path and parameters in **denseOpticalFlow.py** and **sparseOpticalFlow.py**, and run these two files as needed.
4. Post-processing the results via **GetData&DrawFigure.py** and **HVHistograms.py**.
# Main Parameters in Codes
1. Lines 9-19 in **sparseOpticalFlow.py** can be changed as needed, explanation on parameters of the Lucas-Kanade sparse optical flow algorithm can be found in [OpenCV documentation](https://docs.opencv.org/4.9.0/dc/d6b/group__video__track.html). Line 26 and Line 31 are the path of the input video and the name of the output video respectively.
2. Lines 64 in **denseOpticalFlow.py** contains main parameters of the Farnebäck dense optical flow algorithm which can be found in [OpenCV documentation](https://docs.opencv.org/4.9.0/dc/d6b/group__video__track.html). Line 42 and Line 46 are the path of the input video and the name of the output video respectively.
3. Velocity distribution in seconds over the range can be plotted via changing the numbers in Lines 32 of **GetData&DrawFigure.py**. Line 30 is the path of all data derived from sparse optical flow algorithm.
4. Lines 26-27 in **HVHistograms.py** should be change to H channel and V Channel image paths. Drawing parameters are in lines 29-30.
# Understanding the Results
1. Run **sparseOpticalFlow.py** will get a txt file **Alldata.txt** that records all optical flow information, a csv file **optical_flow_energy.csv** that records optical flow energy information, and an **output video**. The output video contain the original image (top left), grayscale converted image (top right), corner detection results (bottom left) and extracted optical flow trajectory (bottom right).
2. Run **denseOpticalFlow.py** will get an **output video**. The output video contain the optical flow vector field visualization (top left), HSV-encoded dense optical flow visualization (top right), H channel visualization (bottom left) and V Channel visualization (bottom right).
# Notes
* The length and width of the output video are both twice the input video. Input video can be scaled down for faster speed but at the expense of accuracy.
* Displacement measurements directly obtained by the optical flow methods are in pixels.
* Analytical results and datasets are available in zenodo [Data From "A Novel Joint Framework of Computer Vision and Seismic Signal Analysis to Detect Deformation and Dynamic Characteristics of Large Rock Avalanches"](https://doi.org/10.5281/zenodo.15615140)
