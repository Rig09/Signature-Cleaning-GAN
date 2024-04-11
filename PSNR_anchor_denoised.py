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
    chi_anchor_dir = "pytorch-CycleGAN-and-pix2pix/datasets/chinesedata/testA"
    eng_anchor_dir = "pytorch-CycleGAN-and-pix2pix/datasets/gan_signdata_kaggle/trainA"
    with open('psnr_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['loss function', 'image_num', 'PSNR_noisy_denoised', 'PSNR_anchor_noisy', 'psnr_anchor_denoised']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()
        chinese = 0
        for sub_dir in sub_directs:
            sub_directory = sub_dir + "/images"
            print(sub_directory)
            files = os.listdir(sub_directory)
            numbers = []
            for file in files:
                num = extract_numbers(file)
                if num and num not in numbers:
                    numbers.append(num)
                if '会' in num or '澉' in num or '学' in num or '荷' in num or '喆' in num or '军' in num or '勤' in num:
                    chinese = 1
            print(chinese)
            for num in numbers:
                img_dir = sub_directory + '/' + num
                noisy = cv2.imread(img_dir + '_real.png') 
                not_noisy = cv2.imread(img_dir+ '_fake.png') 
                if (chinese == 1):
                    anchor = cv2.imread(chi_anchor_dir + '/' + num + '.jpg')
                else:
                    png_file_path = eng_anchor_dir + '/' + num + '.png'
                    if os.path.exists(png_file_path):
                        anchor = cv2.imread(png_file_path)
                    else:
                        # If .png file doesn't exist, try .PNG
                        png_file_path = eng_anchor_dir + '/' + num + '.PNG'
                        if os.path.exists(png_file_path):
                            anchor = cv2.imread(png_file_path)
                        
                value_1 = PSNR(noisy, not_noisy)
                value_2 = PSNR(anchor, noisy)
                value_3 = PSNR(anchor, not_noisy)
                
                # Write to CSV
                # Write to CSV
                writer.writerow({'loss function': sub_dir, 'image_num': num, 'PSNR_noisy_denoised': value_1, 'PSNR_anchor_noisy': value_2, 'psnr_anchor_denoised': value_3})
                print(f"PSNR value (anchor/noisy) is {value_2} dB for signature {img_dir}")
                print(f"PSNR value (anchor/denoised) is {value_3} dB for signature {img_dir}")
            chinese = 0        

if __name__ == "__main__": 
    main()