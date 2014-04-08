function [cxOut, cyOut, radOut, a, b, phi] = blobDetectAffine(im, sigma, scale, thres, alpha)
% Compute blob using LOG filter response and non-maximum suppression
% given an image 'im'. 'sigma' is the standard deviation of the LOG filter.
% 'scale' is a vector of possible scale. 'thres' specifies generic
% thresholding while 'alpha' specifies thresholding using Harris response
%

dx = [-1 0 1; -1 0 1; -1 0 1]; % Derivative masks
dy = dx';

[h, w] = size(im);
nScale = numel(scale);

% 1. Convolve image with scale-normalized Laplacian at several scales
lapResp = zeros(h, w, nScale);
hlogNorm = sigma^2*fspecial('log', floor_odd(6*sigma), sigma);
for l = 1:nScale
    % Downsampling
    tmpIm = imresize(im, round([h,w]/scale(l)), 'bicubic');
    tmpIm = imfilter(tmpIm, hlogNorm, 'symmetric').^2; 
    % Upsampling
    lapResp(:,:,l) = imresize(tmpIm, [h, w], 'bicubic');
end
% 2. Find maxima of squared Laplacian response in scale-space (nonmaximum suppresion)
scale_space = zeros(h, w, nScale);
maxMat = zeros(h, w, nScale);
% Intra-slice nonmaximum suppression
for l = 1:nScale
    % Ensure local max is unique
    maxMat1 = ordfilt2(lapResp(:,:,l), 25, ones(5,5));
    maxMat2 = ordfilt2(lapResp(:,:,l), 24, ones(5,5));
    scale_space(:,:,l) = lapResp(:,:,l).*...
        (lapResp(:,:,l) == maxMat1 & maxMat1 ~= maxMat2);
    maxMat(:,:,l) = maxMat1;
end

% Inter-slice nonmaximum suppresion
scale_space = reshape(scale_space, h*w, nScale);
maxMat = reshape(maxMat, h*w, nScale);
for l = 1:nScale
    nzIdx = find(scale_space(:,l) ~= 0);
    if (l == 1)
        supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l+1));
    elseif (l == nScale)
        supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l-1));
    else
        supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l-1) |...
                      scale_space(nzIdx, l) <= maxMat(nzIdx, l+1));
    end
    scale_space(nzIdx(supIdx), l) = 0;
end
% Thresholding
scale_space(scale_space < thres) = 0;  

% Additional thresholding using Harris response and
% affine adaptation
nzIdx = find(scale_space ~= 0);
[cy, cx, radIdx] = ind2sub([h, w, nScale], nzIdx);
rad = scale(radIdx)'*sigma*sqrt(2);

Ix = conv2(im, dx, 'same');    % Image derivatives
Iy = conv2(im, dy, 'same');
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;

cyOut = -1*ones(numel(nzIdx), 1);
cxOut = -1*ones(numel(nzIdx), 1);
radOut = -1*ones(numel(nzIdx), 1);
lamOut = cell(numel(nzIdx), 1);
for l = 1:numel(rad)
    % Use a Gaussian window that is a factor of 1.5 or 2 larger than the characteristic scale of the blob
    g = fspecial('gaussian', floor_odd(6*rad(l)*2/sqrt(2)), rad(l)*2/sqrt(2));
    M = [imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ix2, g) imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ixy, g);...
         imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ixy, g) imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Iy2, g)];
    R = det(M) - alpha*trace(M)^2; % Harris corner response
    if (R > 0)
        cxOut(l) = cx(l);
        cyOut(l) = cy(l);
        radOut(l) = rad(l);
        [V, D] = eig(M);
        [D, idx] = sort(diag(D)); % Sort eigvec according to eigval
        lamOut{l} = [diag(D), V(:,idx)];
    end
end
cyOut(cyOut == -1) = [];
cxOut(cxOut == -1) = [];
radOut(radOut == -1) = [];
lamOut(cellfun('isempty',lamOut)) = [];

% Compute ellipse's param
a = zeros(numel(lamOut), 1);
b = zeros(numel(lamOut), 1);
phi = zeros(numel(lamOut), 1);
for l = 1:numel(lamOut)
    % Relative shape
    a(l) = lamOut{l}(1,1)^(-1/2); % major axis
    b(l) = lamOut{l}(2,2)^(-1/2); % minor axis
    % Ensure no ambiguity between angle and orientation
    if lamOut{l}(2,3) < 0
        phi(l) = acos(dot(lamOut{l}(:,3),[1; 0])/norm(lamOut{l}(:,3))/norm([1; 0]));
    else
        phi(l) = acos(dot(-lamOut{l}(:,3),[1; 0])/norm(lamOut{l}(:,3))/norm([1; 0]));
    end
    % Absolute scale 
    tmpa = a(l)/(a(l)+b(l))*radOut(l)*2; 
    tmpb = b(l)/(a(l)+b(l))*radOut(l)*2;
    a(l)= tmpa;
    b(l)= tmpb;
end