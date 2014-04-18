% ECE 549 UIUC
% Main script for ECE 549 final project
% Long Le
% University of Illinois
%

clear all; close all;
%run('../../vlfeat/toolbox/vl_setup')

%% Very specific scene for this video.
%{
vid = VideoReader('charade.mp4');
frameIdx = 1;%:500:vid.NumberOfFrames; % frame index
nFrame = numel(frameIdx);
img = cell(1, nFrame);
for k = 1:nFrame
    img{k} = read(vid, frameIdx(k));
end
%}

folder = 'testim';
fnames = dir(fullfile(folder, '*.jpg'));
nFrame = length(fnames);
img = cell(1, nFrame);
for k = 1:nFrame
    img{k} = imread(fullfile(folder, fnames(k).name));
end

%% Feature detection and description
blobSizeThresh = 10;

im = cell(1, nFrame);
f = cell(1, nFrame); % frame
d = cell(1, nFrame); % descriptor
p = cell(1, nFrame); % raw patch
for k = 1:nFrame
    im{k} = single(rgb2gray(img{k})); % Shouldn't use imshow to display this
    [f{k}, d{k}] = vl_covdet(im{k}, 'EstimateOrientation', true, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5);
    [~, p{k}] = vl_covdet(im{k}, 'EstimateOrientation', true, 'EstimateAffineShape', true, 'PeakThreshold', 5, 'EdgeThreshold', 5, 'descriptor', 'Patch');
    
    % Screen out too small blobs
    suppress = false(1, size(f{k}, 2));
    for l = 1:size(f{k}, 2)
        tmp = f{k}(:,l);
        % scales of the affine xformation matrix A, see oriented
        % ellipse and vl_demo_frame.m for details
        tmp = svd([tmp(3) tmp(5); tmp(4) tmp(6)]);
        if (max(tmp) < blobSizeThresh)
            suppress(l) = true;
        end
    end
    f{k}(:, suppress) = [];
    d{k}(:, suppress) = [];
    p{k}(:, suppress) = [];
    
    % Display
    figure; imshow(img{k});
    h = vl_plotframe(f{k});
    set(h,'color','y','linewidth',2);
end

%% Clustering
F = cell2mat(f); % 6 x n
D = cell2mat(d); % 128 x n
P = cell2mat(p); % 1681 x n
% Debugging
%{
for k = 1:10
    figure;
    imPatch = reshape(P(:,k), 41,41);
    imagesc(imPatch);
end
%}

K = 32;
for k = 1:nFrame
    [C, A]= vl_kmeans(double(D), K, 'NumRepetitions', 10);
end
% Debugging
figure; hist(double(A), K);

% Visual words/dictionary
vwordF = cell(1, K);
vwordD = cell(1, K);
vwordP = cell(1, K);
for k = 1:K
    vwordF{k} = F(:, A == k);
    vwordD{k} = D(:, A == k);
    vwordP{k} = P(:, A == k);
end
% Debugging
figure;
for k = 1:size(vwordP{15}, 2)
    imPatch = reshape(vwordP{15}(:,k), 41,41);
    subplot(1, size(vwordP{15}, 2), k); imagesc(imPatch)
end

%% Compute frequency vector for all the frame
for k = 1:nFrame
    
end
