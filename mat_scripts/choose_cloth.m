clear;close all; clc;
%%
file_path='/root/Desktop/GAN/dataset/test_demo_textureGAN/test_demo_fashion2/';
list_path='/root/Desktop/GAN/dataset/picture_2w_cloth/train/';
trainA_path='/root/Desktop/GAN/dataset/test_demo_split2/test_demo_fashion/testA/';
trainB_path='/root/Desktop/GAN/dataset/test_demo_split2/test_demo_fashion/testB/';

img_path_list=dir(strcat(trainA_path,'*.png'));
newim=uint8(zeros(256,256*2,3));
for i = 10:10
    contour_name = strcat(trainA_path,num2str(i),'.png');
    ground_name = strcat(trainB_path,num2str(i),'.png');
    contour = imread(contour_name);
    ground = imread(ground_name);
   
    figure,crop_patch = imcrop(ground);
    [c_h,c_w,~]=size(crop_patch);
    close all;
    imshow(contour);
    [x,y]=ginput(1);
    x=round(x);
    y=round(y);
    plot(x,y,'ro');
    
    contour(y:y+c_h-1,x:x+c_w-1,:)=crop_patch;
    imshow(contour);
    a=1;
    newim(:,1:256,:)=contour;
    newim(:,257:512,:)=ground;
    new_name = strcat(file_path,num2str(i),'.jpg'); 
    imwrite(newim,new_name);
    close all;
end
