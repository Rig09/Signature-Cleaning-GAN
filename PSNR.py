from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
    # Resize the compressed image to match the dimensions of the original image
    compressed_resized = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    mse = np.mean((original - compressed_resized) ** 2) 
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main():
    print(f"anchor and denoised:")
    anchor = cv2.imread("pytorch-CycleGAN-and-pix2pix/datasets/chinesedata/testA/丁勇军-32-3.jpg") 
    denoised = cv2.imread("pytorch-CycleGAN-and-pix2pix/results/gan_signdata/test_latest/images/丁勇军-32-3_fake.png")  
    noisy = cv2.imread("pytorch-CycleGAN-and-pix2pix/results/gan_signdata/test_latest/images/丁勇军-32-3_real.png") 
    value = PSNR(anchor, denoised) 
    print(f"PSNR value is {value} dB for signature 丁勇军-32-3 (denoised/anchor)") 
    value_2 = PSNR(anchor, noisy) 
    print(f"PSNR value is {value_2} dB for signature 丁勇军-32-3 (noisy/anchor)") 
       
if __name__ == "__main__": 
    main()

# from math import log10, sqrt 
# import cv2 
# import numpy as np 
# import os
# import re

# def extract_numbers(file_path):
#     # Extract the file name from the file path
#     file_name = file_path.split("/")[-1]
    
#     # Extract the numbers using regular expression
#     match = re.search(r'([\w\d_-]+)_\w+\.png', file_name)
    
#     if match:
#         return match.group(1)
#     else:
#         return None

  
# def PSNR(original, compressed): 
#     # Resize the compressed image to match the dimensions of the original image
#     compressed_resized = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
#     mse = np.mean((original - compressed_resized) ** 2) 
#     if(mse == 0):  
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse)) 
#     return psnr 
  
# def main():
#     print("Anchor and denoised:")
#     anchor_dir = "pytorch-CycleGAN-and-pix2pix/FEcompare/gan_sign_lk_1/Anchor"
#     denoised_dir = "pytorch-CycleGAN-and-pix2pix/FEcompare/gan_sign_lk_1/ganoutput"

#     anchor_files = os.listdir(anchor_dir)
#     denoised_files = os.listdir(denoised_dir)

#     for anchor_file in anchor_files:
#         # Check if corresponding denoised image exists
#         corresponding_denoised_file = anchor_file.replace(".png", "_fake.png")
#         if corresponding_denoised_file in denoised_files:
#             anchor_path = os.path.join(anchor_dir, anchor_file)
#             denoised_path = os.path.join(denoised_dir, corresponding_denoised_file)

#             anchor = cv2.imread(anchor_path) 
#             denoised = cv2.imread(denoised_path) 

#             value = PSNR(anchor, denoised) 
#             print(f"PSNR value is {value} dB for signature {anchor_file}") 
              
        
# if __name__ == "__main__": 
#     main()
