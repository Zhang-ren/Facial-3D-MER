function [maxrad] = maxflow(onset_save_path,apex_save_path,ox,oy,mode)
% img1 = double(img1);
% img2 = double(img2);
% x = double(x);
% y = double(y);

addpath('mex');

img1 = char(onset_save_path);
img2 = char(apex_save_path);
im1 = double(img1)/255.;
im2 = double(img2)/255.;
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation;

tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;

if mode == 0
    max_ox = int16(max(ox));
    min_ox = int16(min(ox));
    max_oy = int16(max(oy));
    min_oy = int16(min(oy));
    [afflow(:,:,1), afflow(:,:,2)] = affine(flow(:,:,1), flow(:,:,2), ox ,oy);
    u = afflow(min_ox:max_ox,min_oy:max_oy,1);
    v = afflow(min_ox:max_ox,min_oy:max_oy,2);

    
else
    [u, v] = affine(flow(:,:,1), flow(:,:,2), ox ,oy);
end
maxrad = -1;
rad = sqrt(u.^2+v.^2);
maxrad = max(maxrad, max(rad(:)));
