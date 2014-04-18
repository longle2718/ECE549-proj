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

folder = 'imTrain';
file = dir(fullfile(folder, '*.jpg'));
nFrame = length(file);
img = cell(1, nFrame);
for k = 1:nFrame
    img{k} = imread(fullfile(folder, file(k).name));
end

%% Feature extraction for training images
blobSizeThresh = 10;

d = cell(1, nFrame); % descriptor
p = cell(1, nFrame); % raw patch
for k = 1:nFrame
    [d{k}, p{k}] = featExtract(img{k}, blobSizeThresh);
end

%% Clustering to find visual dictionary
D = cell2mat(d);
P = cell2mat(p); 
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
    [C, A]= vl_kmeans(single(D), K, 'NumRepetitions', 10);
end
% Debugging
figure; hist(double(A), K);

% Visual words/dictionary
vwordD = cell(1, K);
vwordP = cell(1, K);
for k = 1:K
    vwordD{k} = D(:, A == k);
    vwordP{k} = P(:, A == k);
end
% Debugging
%{
figure;
for k = 1:size(vwordP{15}, 2)
    imPatch = reshape(vwordP{15}(:,k), 41,41);
    subplot(1, size(vwordP{15}, 2), k); imagesc(imPatch)
end
%}

%% Compute frequency vector for all the frame
% Extract features from a test image
file = dir(fullfile('imTest', '*.jpg'));
imgTest = imread(fullfile('imTest', file.name));
[dTest, pTest] = featExtract(imgTest, blobSizeThresh);

% Compute distance between sift descriptor using L2 norm
freqVec = zeros(1, K);
distMat = vl_alldist2(dTest, C);
