%% Blob detection stuff

sigma = 2;

thresList = [0.03 0.009 0.009 0.003 0.008 0.003 0.005 0.008];
alphaList = [0.18 0.06 0.1 0.06 0.09 0.2 0.18 0.1];

alpha = alphaList(4);
thres = thresList(4);
scale = [1 2 3 4 5 10]/sqrt(2);

vid = VideoReader('charade.mp4');
imgs = cell(1);
samples = 1;

%% Very specific scene for this video.

for sample = 1300: 30:2710
    imgs{samples} = read(vid, sample);
    samples = samples + 1;
end

%% Run blob detection on each of the 48 frame

Cx = cell(1);
Cy = cell(1);
Cr = cell(1);
sifts = cell(1);

descriptors = [];

for i = 1:samples-1
    im = rgb2gray(im2double(imgs{i}));
    [Cx{i}, Cy{i}, Cr{i}, a, b, phi] = blobDetectAffine(im, sigma, scale, thres, alpha);
    sifts{i} = find_sift(im, [Cx{i}, Cy{i}, Cr{i}], 1.5);
    descriptors = [descriptors; sifts{i}];
end

words = cell(1);

%% Work needs to be done to calculate the optimum k.
%% I was thinking about varying it from 4 through 8 or so then
%% use Elbow method to get the best K. For now k = 5.

k = 5;
%% This uses Euclidean distance.
id2clusters = kmeans(descriptors, k);

for i = 1:samples-1
    if i == 1
        c = id2clusters(find(Cx{i}));
    else
        c = id2clusters(find(Cx{i})+size(Cx{i-1},1));
    end
    words{i} = histc(c', linspace(1,k,k));
end