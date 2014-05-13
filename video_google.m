% ECE 549 UIUC
% Search test for ECE 549 final project
% Long Le & Amey Chaugule
% University of Illinois
%

clear all; close all;
%run('../../vlfeat/toolbox/vl_setup')
run('VLFEATROOT/toolbox/vl_setup');
vid = VideoReader('charade.mp4');
imgs = cell(1);
samples = 1;
for sample = 1300: 30:3710
	imgs{samples} = read(vid, sample);
	samples = samples + 1;
end

%% Feature extraction for training images
blobSizeThresh = 10;
nFrame = size(imgs, 2);
d = cell(1, nFrame); % descriptor
p = cell(1, nFrame); % raw patch
for k = 1:nFrame
    [d{k}, p{k}] = featExtract(imgs{k}, blobSizeThresh, false);
end

%% Clustering to find visual dictionary
D = cell2mat(d);
P = cell2mat(p); 

K = 32;
for k = 1:nFrame
    [C, A]= vl_kmeans(single(D), K, 'NumRepetitions', 10);
end
% Debugging
figure; hist(double(A), K);

% Form visual dictionary
vdictD = cell(1, K);
vdictP = cell(1, K);
for k = 1:K
    vdictD{k} = D(:, A == k);
    vdictP{k} = P(:, A == k);
end

words = cell(1, nFrame);
prev_A = 1;
assignments = cell(1);
for k = 1:nFrame
    word = zeros(K,1);
    length = size(d{k},2);
    A_slice = A(prev_A:prev_A+length-1);
    prev_A = length+1;
    for j = 1:K
        word(j) = sum(A_slice==j);
    end
    words{k} = word;
    assignments{k} = A_slice;
    freqVec(k,:) = word;
end

%% Compute frequency vector for all test frame
% Extract features from a test image
imgTest = imgs{30};
[dTest, pTest, fTest] = featExtract(imgTest, blobSizeThresh, true);

figure; imshow(imgTest);
title(sprintf('Choose a part of image to search'));
rect = getrect();
rectangle('Position',rect,'EdgeColor','yellow');

boxF = [];
boxD = [];
index = 1;
word = [];
testA = assignments{30};

for i = 1:size(fTest, 2)
	f = fTest(:,i);
	if (f(1) >= rect(1) && f(1) <= rect(3) && f(2) >= rect(2) && f(2) <= rect(4))
		boxF(:,index) = f;
		boxD(:,i) = dTest(:,i);
        word(index) = testA(i);
        index = index+1;
    else
        boxD(:,i) = dTest(:,i).*0;
	end
end

newFreqVec = [];

for i = 1:nFrame
    freq = [];
    orig_Freq = freqVec(i,:);
    for j = 1:size(word,2)
        freq(j) = orig_Freq(word(j));
    end
    newFreqVec(i,:) = freq;
end

freqVecTest = newFreqVec(30,:);

h = vl_plotframe(boxF);
set(h,'color','y','linewidth',2);

score = vl_alldist2(freqVecTest', newFreqVec', 'HELL');

index = 1;
matches = [];

figure;
for i = 1:size(score, 2)
    if single(score(i)) <= 0.8
        subplot(4,4,index), imshow(imgs{i});
        index = index + 1;
    end
end