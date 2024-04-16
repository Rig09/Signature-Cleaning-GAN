# CycleGAN implementation for Signature Cleaning

The code in this project is implemented to clean signatures with bursty noise from signing on a document. This cleaning would allow existing signature fraud detection systems to have a superior performance on signatures in noisy environments.

###/Outside Technologies used

The project is coded in jupyter notebook and the python programming language. 

The CycleGAN code and paper are used in order to use unpaired data. The cycle GAN original implementation can be seen here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. CycleGAN is a particular architecture of GANs that allows for numerous advantages, icluding the use of unpaired image data. The details of all other advantages and technical background can be found on the original github linked. 

Another tool used by the group is the VGG-16 feature extractor. This was used in the evaluation of the results, to see if denoising signatures increased the difference between real and fraudulent signatures. This was the most important metric used to test the implementation. The VGG-16 is a very deep convolutional network that has won many competitions for its ability to extract key features of images. It is our belief that this could mimic a sophisticated fraud detection system. The metric used after this extraction is cosine simularity. The details and code for this tool can be found here: https://github.com/ashushekar/VGG16

###/ How to implement project code


