function [d, p, f] = featExtract(img, blobSizeThresh, isshow)
% Feature extraction using vl_covdet with blob size 'adjustment'
%
% Input
%   img - input image
% Output
%   d - descriptor  % 128 x n
%   p - raw patch   % 1681 x n
%   f - frame       % 6 x n
%
% Long Le
% University of Illinois
%

im = single(rgb2gray(img)); % Shouldn't use imshow to display this
[f, d] = vl_covdet(im, 'EstimateOrientation', true, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5);
[~, p] = vl_covdet(im, 'EstimateOrientation', true, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5, 'descriptor', 'Patch');

% Screen out too small blobs
suppress = false(1, size(f, 2));
for l = 1:size(f, 2)
    tmp = f(:,l);
    % scales of the affine xformation matrix A, see oriented
    % ellipse and vl_demo_frame.m for details
    tmp = svd([tmp(3) tmp(5); tmp(4) tmp(6)]);
    if (max(tmp) < blobSizeThresh)
        suppress(l) = true;
    end
end
f(:, suppress) = [];
d(:, suppress) = [];
p(:, suppress) = [];

% Display
if isshow
    figure; imshow(img);
    h = vl_plotframe(f);
    set(h,'color','y','linewidth',2);
    title(sprintf('Total %d blobs', size(f, 2)))
end