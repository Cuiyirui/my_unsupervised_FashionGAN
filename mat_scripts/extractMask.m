function [ out_im ] = extractMask( pre_im,noise_thre)
%refine contour
BW2=pre_im<noise_thre;
BW3 = imfill(BW2,'holes');
out_im = BW3;

end

