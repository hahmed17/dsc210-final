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
  
Then, make sure to import the following packages: 

```python
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from tensorly import tt_to_tensor
from tensorly.decomposition import tensor_train
```
### Directions:

To run and check the evaluation of CP Decomposition, run the cells of `cp_decomp_results.ipynb`

To run and check the evaluation of Tensor Train Decomposition, run the cells of `tt_decomp.ipynb`

## Results
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



