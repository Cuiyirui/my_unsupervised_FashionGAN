%%
clear;clc;close all;
mask_thre = 240;
pathA = '/home/Cuiyirui/GAN/dataset/picture_detached/picture_2w_norm/trainA/';
pathB = '/home/Cuiyirui/GAN/dataset/picture_detached/picture_2w_patch/trainA/';
pathC ='/home/Cuiyirui/GAN/dataset/detached_dataset/picture_312_stripe_sequence_patch/testA/';
img_path_list=dir(strcat(pathA,'*.png'));
new_im = zeros(256,512,3);
new_im = uint8(new_im);



%% Extract Mask
for i =1:length(img_path_list)
    im_name = strcat(pathA,num2str(i),'.png');
    im = imread(im_name);
    % extract contour
    contour = im(:,1:256,:);
    contour = rgb2gray(contour);
    % extract mask
    
    mask = extractMask(contour,mask_thre);
    mask = mask*255;
    mask = cat(3,mask,mask,mask);
    % creat new image
    new_im(:,1:256,:)=im;
    new_im(:,257:512,:)=mask;
    new_name = strcat(pathB,num2str(i),'.png');
    imwrite(new_im,new_name);
end
    
   
 %% Extract stripe fashion
%  for i = 1:12
%      stripe_im_name = strcat(pathA,num2str(30+i),'.png');
%      fashion_im_name = strcat(pathB,num2str(i*3),'.png');
%      stripe_im = imread(stripe_im_name);
%      fashion_im = imread(fashion_im_name);
%      stripe_im(:,1:256,:)=fashion_im(:,1:256,:);
%      new_name=strcat(pathA,num2str(i+49),'.png');
%      imwrite(stripe_im,new_name);
%  end


%% rename
% parfor i = 1:length(img_path_list)
%     im_name = strcat(pathA,img_path_list(i).name);
%     %im_name = strcat(pathA,num2str(i),'.jpg');
%     image = imread(im_name);
%     new_name = strcat(pathB,num2str(i),'.png');
%     imwrite(image,new_name)
% end


%% 3 to 2
% for i=1:length(img_path_list)
%     im_name = strcat(pathA,num2str(i),'.jpg');
%     im = imread(im_name);
%     newim = im(:,1:512,:);
%     new_name=strcat(pathB,num2str(i),'.jpg');
%     imwrite(newim,new_name);
%     
% end


%% extract mask and expand dataset
% for i=1:312
%     if(i<10)
%         prefix = 'input_image00';
%     elseif(i>=10 && i<100)
%         prefix = 'input_image0';
%     else
%         prefix = 'input_image';
%     end
%     % define name
%     ground_truth_name = strcat(pathA,prefix,num2str(i),'_encoded.jpg');
%     contour_name = strcat(pathA,prefix,num2str(i),'_ground_truth.jpg');
%     % load data
%     ground_truth = imread(ground_truth_name);
%     contour = imread(contour_name);
%     gray_contour = rgb2gray(contour);
%     % extract mask
%     mask = extractMask(gray_contour,mask_thre);
%     mask = mask*255;
%     mask = cat(3,mask,mask,mask);
%     % creat new image
%     new_im(:,1:256,:)=contour;
%     new_im(:,257:512,:)=ground_truth;
%     new_im(:,513:768,:)=mask;
%     new_name = strcat(pathB,num2str(i),'.png');
%     imwrite(new_im,new_name);
%     % imwrite sample
%     for j = 1:10
%         if(j<10)
%             suffix='_random_sample0';
%         else
%             suffix='_random_sample';
%         end
%         random_sample_name=strcat(pathA,prefix,num2str(i),suffix,num2str(j),'.jpg');
%         random_sample=imread(random_sample_name);
%         new_im(:,1:256,:)=contour;
%         new_im(:,257:512,:)=random_sample;
%         new_im(:,513:768,:)=mask;
%         new_name = strcat(pathB,num2str(i),'_',num2str(j),'.png');
%         imwrite(new_im,new_name);
%     end
%     
% end
%  % bold contour
%  for i =53:64
%      %im_name = strcat(pathA,img_path_list(i).name);
%      im_name =strcat(pathA,num2str(i),'.jpg');
%      im = imread(im_name);
%      % extract contour
%      contour = im(:,1:256,:);
%      contour = rgb2gray(contour);
%      for j=1:256
%          for k=1:256
%              if (contour(j,k)<100)
%                  contour(j,k)=0;
%              end
%          end
%      end
%       contour = cat(3,contour,contour,contour);
%       im(:,1:256,:)=contour;
%       %new_name=strcat(pathB,img_path_list(i).name);
%        new_name =strcat(pathB,num2str(i),'.png');
%       imwrite(im,new_name);
      
 %end
%% normlized line
% parfor i=1:length(img_path_list)
%     %im_name=strcat(pathA,num2str(i),'.jpg');
%     im_name = strcat(pathA,img_path_list(i).name);
%     im = imread(im_name);
%     contour = im(:,1:256,:);
%     %denoise and bold contour
%     contour = rgb2gray(contour);
%     for j=1:256
%         for k=1:256
%             if (contour(j,k)<228)
%                 contour(j,k)=0;
%             end
%         end
%     end
%     contour = im2bw(contour);
%     %thining
%     bw2 = ~bwmorph(~contour,'thin');
%     bw2 = uint8(bw2)*255;
%     contour = cat(3,bw2,bw2,bw2);
%     im(:,1:256,:)=contour;
%     %new_name = strcat(pathB,num2str(i),'.png');
%     new_name = strcat(pathB,img_path_list(i).name);
%     imwrite(im,new_name);
% end

%% detach data
% parfor i=1:length(img_path_list)
%     im_name=strcat(pathA,num2str(i),'.png');
%     im=imread(im_name);
%     contour=im(:,1:256,:);
%     ground=im(:,257:512,:);
%     %write contour image
%     imwrite(contour,strcat(pathB,'trainA/',num2str(i),'.png'));
%     %write pattern image
%      imwrite(ground,strcat(pathB,'trainB/',num2str(i),'.png'));
%     
% end
%% write list file


%% extract 4 data
% new_im = zeros(256,1024,3);
% new_im = uint8(new_im);
% for i = 1:length(img_path_list)
%     im_name=strcat(pathA,num2str(i),'.png');
%     im = imread(im_name);
%     patch_name=strcat(pathB,num2str(i),'.png');
%     patch = imread(patch_name);
%     new_im(:,1:768,:)=im;
%     new_im(:,769:1024,:)=patch;
%     new_name=strcat(pathC,num2str(i),'.png');
%     imwrite(new_im,new_name);
% end

%% cat patch
% new_im = zeros(256,512,3);
% new_im = uint8(new_im);
% for i = 1:length(img_path_list)
%     im_name=strcat(pathA,num2str(i),'.png');
%     im = imread(im_name);
%     patch_name=strcat(pathB,num2str(i),'.png');
%     patch_im = imread(patch_name);
%     patch = patch_im(:,513:768,:);
%     new_im(:,1:256,:)=im;
%     new_im(:,257:512,:)=patch;
%     new_name=strcat(pathC,num2str(i),'.png');
%     imwrite(new_im,new_name);
% end
