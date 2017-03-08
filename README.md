# dSGP: Distributed Sparse Gaussian Process (ICML'16)

CONTENTS OF THIS FILE
---------------------
 * Introduction
 * Requirements
 * Usage
 * Data format
 * Configuration template
 * Disclaimers
 * Maintainers

INTRODUCTION
---------------------
This is the implementation of the following paper: 

A Distributed Variational Inference Framework for Unifying Parallel Sparse Gaussian Process Regression Models.
Trong Nghia Hoang, Quang Minh Hoang and Kian Hsiang Low. 
In Proceedings of the 33rd International Conference on Machine Learning (ICML '16)

REQUIREMENTS
---------------------
1. System:
   - Compatible with Windows/OS X/Linux
     (Tested on Linux server)

2. Supporting softwares:
   - gcc 4.8.1 or above (with OpenMP enabled - refer to: https://gcc.gnu.org/onlinedocs/libgomp/Enabling-OpenMP.html)
   - Armadillo 4.500.0 or above (for installation instruction, refer to: http://arma.sourceforge.net/download.html)

USAGE (FOR LINUX SYSTEM ONLY)
---------------------
1. Compilation:
   g++ *.cpp -o <executable-filename> -O3 -fopenmp -larmadillo -I <armadillo-header-folder-directory>

2. Expected input:
   Raw data in CSV format (unpartitioned, unclustered) (see details below)
   
3. Expected output:
   We produce three types of output:
	1) Real-time statistics (RMSE at different iterations) printed on-screen.
	2) Total incurred time (for constructing the predictive model only) versus number of iterations returned in a .log text file.
	3) Predictive results in terms of root mean square error (RMSE)
  
4. Running:
   The compiled program takes in one single argument which is the configuration file specifying the above expected input and output 
   (see syntax below in CONFIGURATION TEMPLATE section). The deploy command is as followed:
   <executable-filename> <configuration-file>

INPUT FORMAT
---------------------
1.  Raw data format (CSV):
    CSV file, each line contains a tuple of (x, y). The components of x and y are numerical values separated by comma (standard CSV format).

CONFIGURATION TEMPLATE
---------------------
1.  Overall template:
    See example configuration file (aimpeak_config.txt)
	
DISCLAIMERS
---------------------
This code is provided as is. There is no guarantee provided. Please cite our paper if you use it.

MAINTAINERS
---------------------
Quang Minh, Hoang (qminh93@gmail.com)
Trong Nghia, Hoang (hoangtrongnghia87@gmail.com)
