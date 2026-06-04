# SAGAE
# Spatial-Aware Genetic Algorithm Explainer

Because MNIST and Plant Disease Dataset require distinct configuration, two version for SAGAE and MOGAE have been created each well-suited for each of the datasets.

mnist.weights.h5  is the weight of the constructed CNN model for MNIST dataset. You can upload the weights for the sake of reproducability.

Accordingly, Due to massive size of the files (the weights (plantUpsampled4.weights.best.hdf5) of the customized ResNet50 model calculated for reusability, X1_test, y1_test (numpy arrays) are available [here](https://drive.google.com/drive/folders/1_CC8PAPPy9TEaaSVTTfUcEPgWk3O8bXx?usp=sharing) 

The test samples used in the paper for the Plant Disease Dataset are X1_test[5] = Black Spot,  X1_test[29] = Canker,  X1_test[46] = Greening, and X1_test[55] = Healthy.

The test samples used in the paper for the MNIST Dataset are x_test[30] = digit 3, x_test[19] = digit 4, x_test[15] = digit 5, and  x_test[110] = digit 8.

