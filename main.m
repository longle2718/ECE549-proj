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
f = cell(1, nFrame);
d = cell(1, nFrame);
for k = 1:nFrame
    figure; imshow(img{k});
    im = single(rgb2gray(img{k}));
    [f{k}, d{k}] = vl_sift(im, 'WindowSize', 4, 'NormThresh', 5, 'PeakThresh', 5, 'EdgeThresh', 10);
    h = vl_plotframe(f{k});
    set(h,'color','y','linewidth',2);
end

%% Clustering
F = cell2mat(f)';
D = cell2mat(d)';

K = 128;
for k = 1:nFrame
    [idx, C]= kmeans(D, K, 'replicates', 10);
end

word = cell(1, K);
for k = 1:K
    
end