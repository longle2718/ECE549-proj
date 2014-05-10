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
%featExtract(img{k}, blobSizeThresh, true);

%% Tracking using correlation
trackIdx = 0;
%dcumTrack = zeros(128, 0);
%dnumTrack = zeros(1,0);
track = cell(size(d));
track{1} = zeros(1, size(d{1}, 2));
for k = 1:nFrame-1
    track{k+1} = zeros(1, size(d{k+1}, 2));
    
    R = corr(d{k}, d{k+1});
    [idx1, idx2] = ind2sub(size(R), find(R>0.8));
    for l = 1:numel(idx1)
        if track{k}(idx1(l)) == 0
            % Create a new track
            trackIdx = trackIdx + 1;
            track{k}(idx1(l)) = trackIdx;
            track{k+1}(idx2(l)) = trackIdx;
            
            %dcumTrack(:,trackIdx) = d{k}(:, idx1(l)) + d{k+1}(:, idx2(l));
            %dnumTrack(trackIdx) = 2;
        else
            % Continue a track
            track{k+1}(idx2(l)) = track{k}(idx1(l));
            
            %dcumTrack(:, track{k}(idx1(l))) = dcumTrack(:, track{k}(idx1(l))) + d{k+1}(:, idx2(l));
            %dnumTrack(track{k}(idx1(l))) = dnumTrack(track{k}(idx1(l))) + 1;
        end
    end
end

% Debugging
%{
R = corr(d{k}, d{k+1});
[idx1, idx2] = ind2sub(size(R), find(R>0.8));
figure;
subplot(211); imshow(img{k}); hold on; 
for l = 1:numel(idx1)
     text(f{k}(1,idx1(l)), f{k}(2,idx1(l)), num2str(track{k}(idx1(l))), 'color', 'y')
end
subplot(212); imshow(img{k+1}); hold on;
for l = 1:numel(idx2)
     text(f{k+1}(1,idx2(l)), f{k+1}(2,idx2(l)), num2str(track{k+1}(idx2(l))), 'color', 'y')
end
%}
D = cell2mat(d);
%P = cell2mat(p);
TRACK = cell2mat(track);

% Reduce the size of descriptors
D1 = D(:,TRACK == 0);
D2 = zeros(size(D,1), trackIdx);
for k = 1:trackIdx
    D2(:, k) = mean(D(:,TRACK == k), 2);
end
D = [D1 D2];

%% Clustering to find visual dictionary
K = 2^6;
%[C, A]= vl_kmeans(single(D), K, 'NumRepetitions', 10); % using L2 distance
SIGinv = inv(cov(D'));
[C, A]= kmeans_mahal(single(D), K, SIGinv, 10);

% Form visual dictionary
vdictD = cell(1, K);
%vdictP = cell(1, K);
for k = 1:K
    vdictD{k} = D(:, A == k);
    %vdictP{k} = P(:, A == k);
end
%save vdict.mat vdictD

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
cntVec = zeros(nFrame, K);
for k = 1:nFrame
    %distMat = vl_alldist2(d{k}, C); % L2 distance
    distMat = mahal_dist(d{k}, C, SIGinv);
    [~, idx] = min(distMat, [], 2);
    cntVec(k,:) = hist(idx, K);
end
% Create weighted word frequencies
wFreqVec = zeros(size(cntVec));
for k = 1:nFrame
    wFreqVec(k,:) = cntVec(k,:)/sum(cntVec(k,:)).*log(nFrame./sum(cntVec));
end

%save wFreqVec.mat wFreqVec

%% Compute frequency vector for a test frame
% Extract features from a test image
file = dir(fullfile('imTest', '*.jpg'));
imgTest = imread(fullfile('imTest', file(7).name));
[dTest, pTest] = featExtract(imgTest, blobSizeThresh, true);

% Compute distance between sift descriptor
%distMat = vl_alldist2(dTest, C);
distMat = mahal_dist(dTest, C, SIGinv);
[~, idx] = min(distMat, [], 2);
cntVecTest = hist(idx, K);
wFreqVecTest = cntVecTest/sum(cntVecTest).*log(nFrame./sum(cntVec));

xlabel('Visual word');
ylabel('Count');

% Find the most similar image from the training dataset
score = vl_alldist2(cntVecTest', cntVec', 'HELL'); % Hellinger distance for probability measures
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