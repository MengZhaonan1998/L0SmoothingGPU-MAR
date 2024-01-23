# L0SmoothingGPU-MAR
## What is this project for?
This project tries to reduce the metal artifact of CBCT-reconstruction images utilizing L0-smoothing algorithm. The whole project is the CUDA implementation of the paper: http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf , within the context of CT medical imaging applications.

## How to try it out?
The entire project has been developed on Visual Studio, leveraging NVIDIA GPUs through CUDA. Therefore, all you need are:
1. Windows 10/11
2. Visual Studio 2019
3. CUDA 11.3 or above

Clone this repository and simply open the .vcxproj file and build the whole project in VS.
