%%
clear;clc;close all;
%%
file_path='/root/Desktop/GAN/dataset/test_demo/test_demo_stripe/';
list_path='/root/Desktop/GAN/dataset/test_demo_textureGAN/test_demo_groundTruth/';
trainA_path='/root/Desktop/GAN/dataset/test_demo_split2/test_demo_stripe/contour/';
trainB_path='/root/Desktop/GAN/dataset/test_demo_split2/test_demo_stripe/ground/';

img_path_list=dir(strcat(file_path,'*.png'));



%% processing image
% for i=1:24100
%     im_name=strcat(file_path,num2str(i),'.jpg');
%     image=imread(im_name);
%     imA=image(:,1:256,:);
%     imB=image(:,257:512,:);
%     imA_name=strcat(trainA_path,num2str(i),'.jpg');
%     imB_name=strcat(trainB_path,num2str(i),'.jpg');
%     imwrite(imA,imA_name);
%     imwrite(imB,imB_name);
% end

%% processing image
% for i=1:243
%     a=strcat('./',num2str(i),'.jpg');
%     list_name=strcat(list_path,'list_testA.txt');
%     fid=fopen(list_name,'a+');
%     fprintf(fid,'%s\n',a);
%     fclose(fid);
% end

%% process texutureGAN
% for i = 1:length(img_path_list)
%     img_name = strcat(file_path,num2str(i),'.png');
%     img = imread(img_name);
%     ground = img(:,257:512,:);
%     cloth = ground(97:97+63,97:97+63,:);
%     img(97:97+63,97:97+63,:)=cloth;
%     new_name = strcat(list_path,num2str(i),'.png');
%     imwrite(img,new_name);
% end


%% process 3 samples
% newim=uint8(zeros(256,256*3,3));
% for i = 1:1000
%     img_name = strcat(trainA_path,num2str(i),'.jpg');
%     img = imread(img_name);
%     ground = img(:,257:512,:);
%     cloth = ground(97:97+64,97:97+64,:);
%     white_patch=uint8(255*ones(256,256,3));
%     white_patch(end-64:end,end-64:end,:)=cloth;
%     newim(:,1:512,:)=img;
%     newim(:,513:end,:)=white_patch;
%     new_name = strcat(trainB_path,num2str(i),'.jpg');  
%     imwrite(newim,new_name);
% end

%% rename
% for i=1:length(img_path_list)
%     image_name=strcat(file_path,img_path_list(i).name);
%     I=imread(image_name);
%     newName=strcat(file_path,num2str(i),'.png');
%     imwrite(I,newName);
%     delete(image_name);
% end


%% exchange
% for i=0:0
%     im_name=strcat(file_path, num2str(i),'.png');
%     im = imread(im_name);
%     ground = im(:,1:256,:);
%     cloth = im(:,257:512,:);
%     im(:,1:256,:)=cloth;
%     im(:,257:512,:)=ground;
%     imwrite(im,im_name);
% end



%% split image
% for i=1:length(img_path_list)
%     im_name = strcat(file_path, num2str(i),'.png');
%     im = imread(im_name);
%     contour = im(:,1:256,:);
%     ground = im(:,257:512,:);
%     contour_name  = strcat(trainA_path, num2str(i),'.png');
%     ground_name = strcat(trainB_path, num2str(i),'.png');
%     imwrite(contour,contour_name);
%     imwrite(ground,ground_name);
% end

%% extract cloth for cycleGAN
for i = 1:length(img_path_list)
    im_name = strcat(file_path, num2str(i),'.png');
    im = imread(im_name);
    contour = im(:,1:256,:);
    ground = im(:,257:512,:);
    %cloth = ground(96:96+63,96:96+63,:);
    %cloth = imresize(cloth,[256,256]);
    %ground=cloth;
    new_name1 = strcat(trainA_path, num2str(i),'.png');
    new_name2 = strcat(trainB_path, num2str(i),'.png');
    imwrite(contour,new_name1);
    imwrite(ground,new_name2);
end

