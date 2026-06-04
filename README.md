# SAGAE
# Spatial-Aware Genetic Algorithm Explainer

Because the MNIST and Plant Disease datasets require distinct configurations, two versions of SAGAE and MOGAE were created, each well-suited for its respective dataset.

mnist.weights.h5  is the weight of the constructed CNN model for MNIST dataset. You can load the CNN model with the uploaded weights to ensure 100% reproducibility.

Accordingly, Due to massive size of the files (the weights (plantUpsampled4.weights.best.hdf5) of the customized ResNet50 model calculated for reusability, X1_test, y1_test (numpy arrays) are available [here](https://drive.google.com/drive/folders/1_CC8PAPPy9TEaaSVTTfUcEPgWk3O8bXx?usp=sharing) 

The test samples used in the paper for the Plant Disease Dataset are X1_test[5] = Black Spot,  X1_test[29] = Canker,  X1_test[46] = Greening, and X1_test[55] = Healthy.

The test samples used in the paper for the MNIST Dataset are x_test[30] = digit 3, x_test[19] = digit 4, x_test[15] = digit 5, and  x_test[110] = digit 8.

To ensure 100% reproducibility, the LIME-guided initial population has been provided.

For any inquiries regarding the paper or assistance with running the provided source code, please feel free to contact the author at: hn_61@yahoo.com, hossein_nematzadeh@mcbs.edu.om, or hnematzadeh@uma.es.
