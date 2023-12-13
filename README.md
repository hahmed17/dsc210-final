# Final Project: Tensor Decomposition

### Course: DSC 210: Numerical Linear Algebra for Data Science

### Professor: Dr.Tsui-wei Weng
## Instructions

Make sure that the following libraries are installed in python 3 environment:

1. TensorFlow:
   - Library: `tensorflow`
   - Installation: `pip install tensorflow`

2. TensorLy:
   - Library: `tensorly`
   - Installation: `pip install tensorly`

3. Scikit-learn:
   - Library: `sklearn`
   - Installation: `pip install scikit-learn`

4. NumPy:
   - Library: `numpy`
   - Installation: `pip install numpy`

5. Matplotlib:
   - Library: `matplotlib`
   - Installation: `pip install matplotlib`

6. OpenCV:
   - Library: `cv2`
   - Installation: `pip install opencv-python`
  
7. tqdm
   - Library: `tqdm`
   - Installation: `pip install tqdm`
  
Then, make sure to import the following packages: 

```python
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train
from tensorly.decomposition import tucker
from tensorly import tt_to_tensor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
```
### Directions:

To run and check the evaluation of CP Decomposition, run the cells of `cp_decomp_results.ipynb`

To run and check the evaluation of Tensor Train Decomposition, run the cells of `tt_decomp.ipynb`

To run and check the basic latent factor analysis of CP Decomposition, Runtime comparison of CP Decomposition & Tensor Train Decomposition, EDA of CIFAR-10 dataset, or example of image compression using CIFAR-10 dataset, run the cells of `basic_analysis__eda__compression.ipynb`

## Results
### PCA of CP Decomposition:
We generated a 14x5x3 order-3 tensor to represent a 2D RGB image. After applying CP decomposition and perform PCA on the factor matrices representing y-axis and x-axis, we can visualize the number of color blocks along each axis by the number of clusters. The shade of the points indicates the size of the color blocks.

![image](https://github.com/hahmed17/dsc210-final/blob/main/PCA%20of%20y-axis.png)

![image](https://github.com/hahmed17/dsc210-final/blob/main/PCA%20of%20x-axis.png)

### Runtime comparison between CP Decomposition and Tensor Train Decomposition:
We generated random tensors with different orders (2 to 12) or random tensors with different dimensions (2 to 30). Then, we compare the runtime of both decomposition methods on these tensors. In general, as the order/dimension increases, the runtime for both methods both increase exponentially; however, the runtime of Tensor Train Decomposition is consistently lower then CP Decomposition.

![image](https://github.com/hahmed17/dsc210-final/blob/main/runtime%20vs%20order%20for%20tensor%20decomposition.png)

![image](https://github.com/hahmed17/dsc210-final/blob/main/runtime%20vs%20dimension%20for%20tensor%20decomposition.png)

### EDA of CIFAR-10 dataset:
The CIFAR-10 dataset has 60,000 images, with 50,000 images in the training set, and 10,000 images in the test set. The dimension of each image is 32x32 pixels, with RGB channels. There are 10 classes in total. The following are 4 sample images in their original size (32x32) and after being downsized (8x8).

![image](https://github.com/hahmed17/dsc210-final/blob/main/cifar10%20normalized.png)

![image](https://github.com/hahmed17/dsc210-final/blob/main/cifar10%20normalized%20resized.png)

### CP Decomposition:
We then tracked changes in performance across four more iterations, each time resizing the CIFAR image data and running CP Decomposition. In total, we measured the performance of TTD on images of dimensions 32x32, 16x16, 8x8, 4x4, 2x2, and 1x1. Below are plots that reflect runtime and CPU usage:

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/5fb1690e-0015-47f6-9d97-0b83051aa389)

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/745fd66f-413d-4aaa-8adb-1a5506eb5b84)

Each of the five times that we resized and preprocessing the data with CP Decomposition, we trained and tested a logistic regression model, which performed with the following accuracy:

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/3749f2b8-411e-4441-9045-43fef290aa4a)

### TT Decomposition: 
We then tracked changes in performance across four more iterations, each time resizing the CIFAR image data and running TT Decomposition. In total, we measured the performance of TTD on images of dimensions 32x32, 16x16, 8x8, 4x4, 2x2, and 1x1. Below are plots that reflect runtime and CPU usage:

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/4aa69466-2745-49ba-baf1-8859d50ef1d3)

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/b87291c0-412e-4f55-8216-663668446157)

Each of the five times that we resized and preprocessing the data with TT Decomposition, we trained and tested a logistic regression model, which performed with the following accuracy:

![image](https://github.com/hahmed17/dsc210-final/assets/56313938/ccacb91c-d2d7-4ea7-9ead-33474cd70b81)

### Image compression:
We used the same dataset to demonstrate the application of tensor decomposition on image compression. An example of the reconstruction results comparison is as followed:

![image](https://github.com/hahmed17/dsc210-final/blob/main/reconstruction%20results%20for%20image%20no.%201.png)

The first image is the original image, the three images in between are the reconstruction results with similar number of parameters stored, and the last image is a reconstruction with similar quality as the original image.

The runtime for each operation is as followed:

![image](https://github.com/hahmed17/dsc210-final/blob/main/runtime%20for%20image%20compression.png)


