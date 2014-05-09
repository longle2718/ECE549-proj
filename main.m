% ECE 549 UIUC
% Main script for ECE 549 final project
% Long Le
% University of Illinois
%

clear all; close all;
%run('../../vlfeat/toolbox/vl_setup')

%% Very specific scene for this video.
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
f = cell(1, nFrame); % frame
for k = 1:nFrame
    [d{k}, p{k}, f{k}] = featExtract(img{k}, blobSizeThresh, false);
end

%% Tracking using correlation
trackIdx = 0;
track = cell(0);
for k = 1:nFrame-1
    R = corr(d{k}, d{k+1});
    [idx1, idx2] = ind2sub(size(R), find(R>0.8));
    % Debugging
    figure;
    subplot(211); imshow(img{k}); hold on; 
    for l = 1:numel(idx1)
         text(f{k}(1,idx1(l)), f{k}(2,idx1(l)), num2str(l), 'color', 'y')
    end
    subplot(212); imshow(img{k+1}); hold on;
    for l = 1:numel(idx2)
         text(f{k+1}(1,idx2(l)), f{k+1}(2,idx2(l)), num2str(l), 'color', 'y')
    end
end

D = cell2mat(d);
P = cell2mat(p); 
ndesc = size(D, 2); % total number of descriptors.

%% Clustering to find visual dictionary
K = 32;
for k = 1:nFrame
    [C, A]= vl_kmeans(single(D), K, 'NumRepetitions', 10);
end

% Form visual dictionary
vdictD = cell(1, K);
vdictP = cell(1, K);
for k = 1:K
    vdictD{k} = D(:, A == k);
    vdictP{k} = P(:, A == k);
end
save vdict.mat vdictD vdictP

% Debugging
%{
figure; hist(double(A), K);
figure;
for k = 1:5%size(vdictP{15}, 2)
    imPatch = reshape(vdictP{1}(:,k), 41,41);
    subplot(1, 5, k); imagesc(imPatch)
end
%}

%% Compute frequency vectors for all training frames
freqVec = zeros(nFrame, K);
for k = 1:nFrame
    distMat = vl_alldist2(d{k}, C);
    [~, idx] = min(distMat, [], 2);
    freqVec(k,:) = hist(idx, K);
end
save freqVec.mat freqVec

%% Compute frequency vector for a test frame
% Extract features from a test image
file = dir(fullfile('imTest', '*.jpg'));
imgTest = imread(fullfile('imTest', file(2).name));
[dTest, pTest] = featExtract(imgTest, blobSizeThresh, true);

% Compute distance between sift descriptor using L2 norm
distMat = vl_alldist2(dTest, C);
[~, idx] = min(distMat, [], 2);
freqVecTest = hist(idx, K);
xlabel('Visual word');
ylabel('Count');

% Find the most similar image from the training dataset
score = vl_alldist2(freqVecTest', freqVec', 'HELL');
[sortScore, frameIdx] = sort(score);

% Display the top N most similar images
N = 5;
figure;
set(gcf, 'units','normalized', 'position', [0 0 1 1])
for k = 1:N
    subplot(1,N,k); imshow(img{frameIdx(k)})
    xlabel(sprintf('Relevance: %0.1f, Frame index: %d', 100-sortScore(k), frameIdx(k)))
    %set(get(gca,'YLabel'),'Rotation',0)
end
