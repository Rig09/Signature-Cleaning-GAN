# CycleGAN implementation for Signature Cleaning

The code in this project is implemented to clean signatures with bursty noise from signing on a document. This cleaning would allow existing signature fraud detection systems to have a superior performance on signatures in noisy environments. An example of the final implementation can be seen Below:

![02_062_real](https://github.com/Rig09/Signature-Cleaning-GAN/assets/128671428/ff789555-ecf0-48e2-9b66-d6f4f2a2d1e9)

![02_062_fake](https://github.com/Rig09/Signature-Cleaning-GAN/assets/128671428/a37b7e39-ee02-4c20-9c63-f8f42da0fae4)

### Outside Technologies used

The project is coded in jupyter notebook and the python programming language. 

The CycleGAN code and paper are used in order to use unpaired data. The cycle GAN original implementation can be seen here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. CycleGAN is a particular architecture of GANs that allows for numerous advantages, icluding the use of unpaired image data. The details of all other advantages and technical background can be found on the original github linked. 

Another tool used by the group is the VGG-16 feature extractor. This was used in the evaluation of the results, to see if denoising signatures increased the difference between real and fraudulent signatures. This was the most important metric used to test the implementation. The VGG-16 is a very deep convolutional network that has won many competitions for its ability to extract key features of images. It is our belief that this could mimic a sophisticated fraud detection system. The metric used after this extraction is cosine simularity. The details and code for this tool can be found here: https://github.com/ashushekar/VGG16

### How to implement project code
To implement the code the updated CycleGAN.ipnyb is run to generate a model. This can be used with a variety of loss functions. Through expirementation, it has been determined that the best loss function to use is LS loss function. So it is recomended to implement that function. Once that code has been run, a model should be generated, the GAN that we want to take advamtage of for our implementation is the A to B GAN. This GAN maps noisy signatures to clean ones. If all files are downloaded this model should sufficiently map the images as show. 

