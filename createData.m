% Script to create train and test images for preliminary test.
%
% Long Le
% longle1@illinois.edu
% University of Illinois
%

clear all; close all;
vid = VideoReader('charade.mp4');
%% Training data
frameIdx = vid.FrameRate/2:vid.FrameRate:vid.NumberOfFrames; % frame index
nFrame = numel(frameIdx);
img = cell(1, nFrame);
for k = 1:nFrame
    img{k} = read(vid, frameIdx(k));
    imwrite(img{k},['imTrain/' num2str(k) '.jpg'], 'jpg');
end

%% Test data
frameIdx = randi(vid.NumberOfFrames, 1, 10);
nFrame = numel(frameIdx);
for k = 1:nFrame
    img{k} = read(vid, frameIdx(k));
    imwrite(img{k},['imTest/' num2str(k) '.jpg'], 'jpg');
end
