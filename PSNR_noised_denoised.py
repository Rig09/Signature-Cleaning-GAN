from math import log10, sqrt 
import cv2 
import numpy as np 
import os
import re
import csv

def extract_numbers(file_path):
    # Extract the file name from the file path
    file_name = file_path.split("/")[-1]
    
    # Extract the numbers using regular expression
    match = re.search(r'([\w\d_-]+)_\w+\.png', file_name)
    
    if match:
        return match.group(1)
    else:
        return None
  
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
    print("Noisy and denoised:")
    dir = "pytorch-CycleGAN-and-pix2pix/results"
    sub_directs = {"pytorch-CycleGAN-and-pix2pix/results/gan_schin_lk_1/gan_lk_1/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_schin_ls/gan_signdata_ls/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_schin_wgp/gan_gp_400/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_sign_lk/gan_lk/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_sign_lk_1/gan_lk_1/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_sign_lk_4/gan_lk_4/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_signdata/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_signed/gan_signdata_wp/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/gan_signed_ls/gan_signdata_ls/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/vgan_chin/vgan_sign/test_latest",
                   "pytorch-CycleGAN-and-pix2pix/results/vgan_sign/vgan_sign/test_latest"}
    with open('psnr_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['loss function', 'image_num', 'PSNR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()
        
        for sub_dir in sub_directs:
            sub_directory = sub_dir + "/images"
            files = os.listdir(sub_directory)
            numbers = []
            for file in files:
                num = extract_numbers(file)
                if num and num not in numbers:
                    numbers.append(num)
            
            for num in numbers:
                noisy_dir = sub_directory + '/' + num
                noisy = cv2.imread(noisy_dir + '_real.png') 
                not_noisy = cv2.imread(sub_directory + '/' + num + '_fake.png') 
                value = PSNR(noisy, not_noisy)
                
                # Write to CSV
                writer.writerow({'sub_direct': sub_dir, 'num': num, 'psnr_value': value})
                print(f"PSNR value is {value} dB for signature {noisy_dir}")        

if __name__ == "__main__": 
    main()