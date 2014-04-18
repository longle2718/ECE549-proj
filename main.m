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
frameIdx = 1:500:vid.NumberOfFrames; % frame index
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

%% Run blob detection on each of the 48 frame
im = cell(1, nFrame);
f = cell(1, nFrame);
d = cell(1, nFrame);
for k = 1:nFrame
    im{k} = single(rgb2gray(img{k}));
    [f{k}, d{k}] = vl_sift(im, 'WindowSize', 4, 'NormThresh', 5, 'PeakThresh', 5, 'EdgeThresh', 10);
    %figure; imshow(img{k});
    %h = vl_plotframe(f{k});
    %set(h,'color','y','linewidth',2);
end

%% Clustering
F = cell2mat(f);
D = double(cell2mat(d));

K = 32;
for k = 1:nFrame
    [C, A]= vl_kmeans(D, K, 'NumRepetitions', 10);
end

vword = cell(1, K); % Visual words/dictionary
for k = 1:K
    vword{k} = F(:, A == k);
end

% Debug by plotting all the image patches
vword

%% Compute frequency vector for all the frame
for k = 1:nFrame
    
end
