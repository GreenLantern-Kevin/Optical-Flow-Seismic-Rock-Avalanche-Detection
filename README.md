# Optical-Flow-Seismic-Rock-Avalanche-Detection
Optical-Flow-Seismic-Rock-Avalanche-Detection is a package to detect deformation and dynamic characteristics with computer vision and seismic signal analysis. The manuscript "A Novel Joint Framework of Computer Vision and Seismic Signal Analysis to Detect Deformation and Dynamic Characteristics of Large Rock Avalanches" has been submitted to Geophysical Research Letters. Once the manuscript is accepted and published, the paper will be uploaded in the repository and the repository will be updated.
# Recommended Interpreter
* Python 3.10 with packages opencv-python (4.9.0), matplotlib (3.9.0), numpy (1.26.4), pandas (2.2.2), cvzone (1.6.1), sympy (1.12) installed
* Windows 10/11
# Description of Files
* **denseOpticalFlow.py** and **sparseOpticalFlow.py** are fundamental
* config.yml references the file paths to the data and is called by the python script.
TrackFlow contains the code. runTrackFlow.py executes the program in the command line. The functions folder contains the helper functions.
1_Data contain the data necessary to reproduce the results in the paper, including the DSMs, Imagery, outputs from CIAS, and the manually derived validation vectors.
config.yml is the config file used to run runTrackFlow.py. Edit this file to run the code.
# Instructions
Download TrackFlow and 1_Data folders and the config.yml file and put into an overarching folder somewhere in your file system.
Ensure that required Python packages are installed
Edit the config.yml file. Edit the windows and/or linux drivemaps to match your system. Edit the project directory path to the location of the overarching folder.
Run runTrackFlow.py in Python command line. The file takes one argument, "-c", which specifies the path to the config.yml file.
e.g. "python path_to_code/runTrackFlow.py -c path_to_config/config.yml"

# Results Folders
2_Optical_Flow_Results is where the optical flow outputs are stored for the whole area of interest (Geotiff) and validation points (csv).
3_CIAS_Results is where the averaged CIAS results are stored as well as the interpolations of the sparse grid ouputs from CIAS.
4_Validation is where the summary tables for the validation vectors are stored.
5_Plots is where the output plots are stored.
# Understanding the Results
2_Optical_Flow_Results has the results from the optical flow algorithm stored as a 9 or 11 band geotiff. Bands are as follows:
θ - direction angle of the displacement vector for each pixel
l - magnitude of the displacement vector
U - x-direction component of the displacement vector
V - y-direction component of the displacement vector
σ_d - standard deviation of the angle of displacement
σ_m - standard deviation of the displacement magnitude
error band - error band constructed from σ_d, σ_m, and l
σ_u - standard deviation of the u component
σ_v - standard deviation of the v component
u_filt - u component for masked vectors (unreliable vectors removed)
v_filt - v component for masked vectors (unreliable vectors removed)
validation folder within 2_Optical_Flow_Results contains the optical flow predictions for the manually derived displacement vectors (csvs). These predictions are made for the different datasets (ortho, hillshade, 24 hillshade pairs, 168 hillshade pairs) with or without filtering schemes applied. These files have the starting X,Y coordinate and ending X,Y coordinate of each prediction vector as well as the u- and v- components.
3_CIAS_Results contains two folders, fullfield has the interpolated results from CIAS over the whole area of interest, validation has the predictions for the validation vectors for different datasets (avg of 24 hillshades, 1 hillshade, ortho, avg of 24 hillshade filtered) as csvs.
The fullfield geotiffs have 3 bands:
U - x-direction component of the displacement vector
V - y-direction component of the displacement vector
Cross Correlation Coefficient - NCC coefficient corresponding to the prediction
The validation sets have the X,Y starting coordinates, dx, dy which are the U,V components, length and direction of the vectors, the NCC coefficient, and the error band & log(error band) (average hillshade sets only) for each vector. dx, dy = -9999 if vectors are filtered by the hillshade scheme.
4_Validation contain the tables summarising the comparisons between the manually derived vectors and the predictions for various datasets for the NCC and the Optical Flow tracking methods
5_Plots contain the summary plots illustrating selected results that are included in the paper.
# Notes
All imagery is georeferenced using New Zealand Transverse Mercator projection (NZTM, EPSG: 2193), all elevations are height above ellipsoid
Displacement measurements are in metres
