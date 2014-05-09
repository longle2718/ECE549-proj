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
[f, d] = vl_covdet(im, 'EstimateOrientation', false, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5);
[~, p] = vl_covdet(im, 'EstimateOrientation', false, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5, 'descriptor', 'Patch');

% Screen out too small blobs
suppress = false(1, size(f, 2));
%uf = zeros(5, size(f, 2)); % unoriented frame
for l = 1:size(f, 2)
    tmp = f(:,l);
    A = [tmp(3) tmp(5); tmp(4) tmp(6)];
    % scales of the affine xformation matrix A, see oriented
    % ellipse and vl_demo_frame.m for details
    tmp = svd(A);
    if (max(tmp) < blobSizeThresh)
        suppress(l) = true;
    end
    % xform from oriented to nonoriented ellipse
    %S = A*A';
    %uf(:,l) = [f(1:2,l); S(1,1) ; S(1,2) ; S(2,2)];
end
%f = uf;
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