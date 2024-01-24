# L0SmoothingGPU-MAR
## What is this project about?
This project tries to reduce the metal artifact of CBCT-reconstruction images utilizing L0-smoothing algorithm. The whole project is the CUDA implementation of the algorithm presented by paper: http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf , within the context of CT medical imaging applications.

## What does this project contain?
There are three directories:
1. TestData: Two .raw test data, one of which is a single-slice image while the another one is a multi-slice (10 slices) version. They serve as the input data of the algorithm and you may view them through ImageJ.
2. matlab: A matlab implementation of the algorithm.
3. src: Source codes of the C implementation for the algorithm.

## How to try it out?
The C implementation has been developed on Visual Studio 2019, leveraging NVIDIA GPUs through CUDA. Therefore, all you need are:
1. Windows 10
2. Visual Studio 2019
3. CUDA 11.3

Clone this repository and simply open the .vcxproj file and build the whole project in VS.

## A sample of CBCT imaging
### What is CBCT?
CBCT (Cone-Beam Computed Tomography) reconstruction is a process that creates tomographic images from X-ray projection data. 
### What is MAR in the repo name for?
Metal artifact is one of the artifacts commonly encountered in clinical CT imaging and may obscure pathology. There have been various methods proposed to ruduce metal artifacts in CT imaging. This project is exactly an another try on MAR (Metal Artifact Reduction). 
### How does L0SmoothingGPU work on MAR?
L0SmoothingGPU takes a single/multi-slice raw data of a CBCT image as a input and outputs the raw data after reducing metal artifacts. The figures above depict the effect of L0Smoothing, where the left-hand side represents the original image, and the right-hand side signifies the output after MAR.

<p align="center">
  <img src='TestData/test_recon_768_768.jpg' width='350'>    
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src='TestData/output_test_recon_768_768.jpg' width='350'>  
</p>

## Performance measurement
The project heavily uses CUDA to accelerate the running speed of the algorithm. It is worth noting that it is not very efficient to process only a single slice on GPU through CUDA due to the device-launch latency (even much slower than CPU processing). However, when it comes to multi-slice processing, the CUDA implementation is quite powerful. In main.cpp, there are two demos which can prove this statement: It takes hundreds of milliseconds (around 700 ms) to process the first slice and afterwards it takes less than 10 milliseconds in average to process every slice (on RTX 3060 Laptop).  
