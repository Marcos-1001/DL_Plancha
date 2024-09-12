# https://anushsom.medium.com/image-augmentation-for-creating-datasets-using-pytorch-for-dummies-by-a-dummy-a7c2b08c5bcb
import PIL
import torch 
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torchvision.transforms as T
import os

#torch.transforms

#grayscale
grayscale_transform = T.Grayscale(3)

#random rotation
random_rotation_transformation_45 = T.RandomRotation(45)
random_rotation_transformation_85 = T.RandomRotation(85)
random_rotation_transformation_65 = T.RandomRotation(65)

#Gaussian Blur
gausian_blur_transformation_13 = T.GaussianBlur(kernel_size = (7,13), sigma = (6 , 9))
gausian_blur_transformation_56 = T.GaussianBlur(kernel_size = (7,13), sigma = (5 , 8))

#Gaussian Noise

def addnoise(input_image, noise_factor = 0.3):
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip (noisy,0,1.)
    output_image = T.ToPILImage()
    image = output_image(noisy)
    return image

#Colour Jitter

colour_jitter_transformation_1 = T.ColorJitter(brightness=(0.5,1.5),contrast=(3),saturation=(0.3,1.5),hue=(-0.1,0.1))

colour_jitter_transformation_2 = T.ColorJitter(brightness=(0.7),contrast=(6),saturation=(0.9),hue=(-0.1,0.1))

colour_jitter_transformation_3 = T.ColorJitter(brightness=(0.5,1.5),contrast=(2),saturation=(1.4),hue=(-0.1,0.5))

#Random invert

random_invert_transform = T.RandomInvert()

#Main function that calls all the above functions to create 11 augmented images from one image

def augment_image(img_path):

    #orig_image
    orig_img = Image.open(Path(img_path))

    #grayscale
    
    grayscaled_image=grayscale_transform(orig_img)
    #grayscaled_image.show()
    
    #random rotation
    random_rotation_transformation_45_image=random_rotation_transformation_45(orig_img)
    #random_rotation_transformation_45_image.show()
    
    random_rotation_transformation_85_image=random_rotation_transformation_85(orig_img)
    #random_rotation_transformation_85_image.show()
    
    random_rotation_transformation_65_image=random_rotation_transformation_65(orig_img)
    #random_rotation_transformation_65_image.show()
    
    #Gausian Blur
    
    gausian_blurred_image_13_image = gausian_blur_transformation_13(orig_img)
    #gausian_blurred_image_13_image.show()

    gausian_blurred_image_56_image = gausian_blur_transformation_56(orig_img)
    #gausian_blurred_image_56_image.show()
    
    #Gausian Noise

    gausian_image_3 = addnoise(orig_img)
    
    #gausian_image_3.show()

    gausian_image_6 = addnoise(orig_img,0.6)
    
    #gausian_image_6.show()
    
    gausian_image_9 = addnoise(orig_img,0.9)

    #gausian_image_9.show()

    #Color Jitter

    
    colour_jitter_image_1 = colour_jitter_transformation_1(orig_img)
    
    #colour_jitter_image_1.show()
    
    
    colour_jitter_image_2 = colour_jitter_transformation_2(orig_img)
    
    #colour_jitter_image_2.show()
    
    colour_jitter_image_3 = colour_jitter_transformation_3(orig_img)

    #colour_jitter_image_3.show()

    return [orig_img,grayscaled_image,random_rotation_transformation_45_image,random_rotation_transformation_65_image,random_rotation_transformation_85_image,gausian_blurred_image_13_image,gausian_blurred_image_56_image,gausian_image_3,gausian_image_6,gausian_image_9,colour_jitter_image_1,colour_jitter_image_2,colour_jitter_image_3]

#augmented_images = augment_image(orig_img_path)

def creating_file_with_augmented_images(folder_path ):

    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            images = augment_image(folder_path+"/"+file)
            for i in range(len(images)):
                images[i].save(folder_path+"/"+file[:-4]+"_augmented_"+str(i)+".jpg")
    
    


#augmented dataset path
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# master dataset path
for i in range(6):
    folder_path = "data/seg_train/seg_train/"+classes[i]+"/"    
    creating_file_with_augmented_images(folder_path)


# run the program
