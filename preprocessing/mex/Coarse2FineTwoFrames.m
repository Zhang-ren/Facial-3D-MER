function [vx, vy, warpI2] = Coarse2FineTwoFrames(im1, im2, para)
%COARSE2FINETWOFRAMES Estimate optical flow between two frames.
%   [vx, vy, warpI2] = Coarse2FineTwoFrames(im1, im2, para) keeps the
%   original project call signature while providing a MATLAB function
%   implementation. This avoids the "script cannot be called as a function"
%   error when MATLAB resolves Coarse2FineTwoFrames.m from the mex path.

if nargin < 3 || isempty(para)
    para = [0.012, 0.75, 20, 7, 1, 30];
end

alpha = para(1);
ratio = para(2);
minWidth = para(3);
nOuterFPIterations = para(4);
nInnerFPIterations = para(5);
nSORIterations = para(6);

im1 = im2double(im1);
im2 = im2double(im2);

[pyr1, pyr2] = buildPyramid(im1, im2, ratio, minWidth);
vx = zeros(size(pyr1{end}, 1), size(pyr1{end}, 2));
vy = zeros(size(pyr1{end}, 1), size(pyr1{end}, 2));

for level = numel(pyr1):-1:1
    I1 = pyr1{level};
    I2 = pyr2{level};

    if level < numel(pyr1)
        targetSize = [size(I1, 1), size(I1, 2)];
        scaleY = targetSize(1) / size(vx, 1);
        scaleX = targetSize(2) / size(vx, 2);
        vx = imresize(vx, targetSize, 'bilinear') * scaleX;
        vy = imresize(vy, targetSize, 'bilinear') * scaleY;
    end

    I1gray = toGray(I1);
    for outer = 1:nOuterFPIterations
        I2warp = warpImage(I2, vx, vy);
        I2gray = toGray(I2warp);
        [du, dv] = hornSchunckIncrement(I1gray, I2gray, alpha, ...
            nInnerFPIterations, nSORIterations);
        vx = vx + du;
        vy = vy + dv;
    end
end

warpI2 = warpImage(im2, vx, vy);
end

function [pyr1, pyr2] = buildPyramid(im1, im2, ratio, minWidth)
pyr1 = {im1};
pyr2 = {im2};

while min(size(pyr1{end}, 1), size(pyr1{end}, 2)) * ratio >= minWidth
    nextSize = max(round([size(pyr1{end}, 1), size(pyr1{end}, 2)] * ratio), [2, 2]);
    pyr1{end + 1} = imresize(pyr1{end}, nextSize, 'bilinear'); %#ok<AGROW>
    pyr2{end + 1} = imresize(pyr2{end}, nextSize, 'bilinear'); %#ok<AGROW>
end
end

function gray = toGray(im)
if ndims(im) == 3
    gray = 0.2989 * im(:, :, 1) + 0.5870 * im(:, :, 2) + 0.1140 * im(:, :, 3);
else
    gray = im;
end
end

function [du, dv] = hornSchunckIncrement(I1, I2, alpha, innerIterations, sorIterations)
du = zeros(size(I1));
dv = zeros(size(I1));

kernelX = 0.25 * [-1, 1; -1, 1];
kernelY = 0.25 * [-1, -1; 1, 1];
kernelT = 0.25 * ones(2);

Ix = conv2(I1, kernelX, 'same') + conv2(I2, kernelX, 'same');
Iy = conv2(I1, kernelY, 'same') + conv2(I2, kernelY, 'same');
It = conv2(I2, kernelT, 'same') - conv2(I1, kernelT, 'same');

avgKernel = [1 2 1; 2 0 2; 1 2 1] / 12;
iterations = max(1, innerIterations) * max(1, sorIterations);

for iter = 1:iterations
    uAvg = conv2(du, avgKernel, 'same');
    vAvg = conv2(dv, avgKernel, 'same');
    der = Ix .* uAvg + Iy .* vAvg + It;
    denom = alpha^2 + Ix.^2 + Iy.^2;
    du = uAvg - Ix .* der ./ denom;
    dv = vAvg - Iy .* der ./ denom;
end
end

function warped = warpImage(im, vx, vy)
[height, width, channels] = size(im);
[x, y] = meshgrid(1:width, 1:height);
queryX = x + vx;
queryY = y + vy;

warped = zeros(size(im), class(im));
for channel = 1:channels
    warped(:, :, channel) = interp2(x, y, im(:, :, channel), queryX, queryY, ...
        'linear', 0);
end
end
