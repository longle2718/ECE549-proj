% ECE 549 UIUC
% Test blob detector with affine transformation algorithm
% Long Le
% longle1@illinois.edu
%
clear all;close all;

% Different implementation methods
convMethod = 2;
nonmaxsupMethod = 3;

% Read images
folder = 'testim'
fnames = dir(fullfile(folder, '*.jpg'));
numfids = length(fnames);
im = cell(1, numfids);

sigma = 2;
dx = [-1 0 1; -1 0 1; -1 0 1]; % Derivative masks
dy = dx';

scaleList = cell(1,numfids);
scaleList{1} = [1 3 5 10 12 16]/sqrt(2);
scaleList{2} = [1 3 5 7]/sqrt(2);
scaleList{3} = [1 3 4 6 10 16 25]/sqrt(2);
scaleList{4} = [1 2 3 4 5 10]/sqrt(2);
scaleList{5} = [1 2 6 10 12 19 22]/sqrt(2);
scaleList{6} = [1 3 5 8 16 25 32]/sqrt(2);
scaleList{7} = [1 3 5 8 16 25]/sqrt(2);
scaleList{8} = [1 3 5 8 16 24]/sqrt(2);
thresList = [0.03 0.009 0.009 0.003 0.008 0.003 0.005 0.008];
alphaList = [0.18 0.06 0.1 0.06 0.09 0.2 0.18 0.1];

for k = 4
    scale = scaleList{k};
    nScale = numel(scale);
    thres = thresList(k);
    alpha = alphaList(k);
    
    fgDbg = figure;
    set(fgDbg, 'units','normalized', 'position', [0 0 1 1])
    fgOut = figure;
	im{k} = imread(fullfile(folder, fnames(k).name));
    im{k} = rgb2gray(im2double(im{k}));
    [h, w] = size(im{k});
    
    % 1. Convolve image with scale-normalized Laplacian at several scales
    figure(fgDbg); colormap(gray);
    lapResp = zeros(h, w, nScale);
    if (convMethod == 1)
        % Scale filters approach
        tic
        for l = 1:nScale
            % Ensure that filter length is odd for peaks detection
            scaledSigma = scale(l)*sigma;
            hlogNorm = scaledSigma^2*fspecial('log', floor_odd(6*scaledSigma), scaledSigma);
            % Squared response to find both local max and min,
            % could use absolute as well
            lapResp(:,:,l) = imfilter(im{k}, hlogNorm, 'symmetric').^2; 
            subplot(2, nScale, l); imagesc(lapResp(:,:,l));
        end
        toc
    else
        % Resample images approach
        tic
        hlogNorm = sigma^2*fspecial('log', floor_odd(6*sigma), sigma);
        for l = 1:nScale
            % Downsampling
            tmpIm = imresize(im{k}, round([h,w]/scale(l)), 'bicubic');
            tmpIm = imfilter(tmpIm, hlogNorm, 'symmetric').^2; 
            % Upsampling
            lapResp(:,:,l) = imresize(tmpIm, [h, w], 'bicubic');
            subplot(2, nScale, l); imagesc(lapResp(:,:,l));
        end
        toc
    end

    % 2. Find maxima of squared Laplacian response in scale-space (nonmaximum suppresion)
    scale_space = zeros(h, w, nScale);
    maxMat = zeros(h, w, nScale);
    % Intra-slice nonmaximum suppression
    if (nonmaxsupMethod == 1)
        % nlfilter
        nlfunc1 = @(x) ordstat(x(:), 9);
        nlfunc2 = @(x) ordstat(x(:), 8);
        tic
        for l = 1:nScale
            maxMat1 = nlfilter(lapResp(:,:,l), [5 5], nlfunc1);
            maxMat2 = nlfilter(lapResp(:,:,l), [5 5], nlfunc2);
            scale_space(:,:,l) = lapResp(:,:,l).*...
                (lapResp(:,:,l) == maxMat1 & maxMat1 ~= maxMat2);
            maxMat(:,:,l) = maxMat1;
        end
        toc
    elseif (nonmaxsupMethod == 2)
        % colfilt
        colfunc1 = @(x) ordstat(x, 9);
        colfunc2 = @(x) ordstat(x, 8);
        tic
        for l = 1:nScale
            maxMat1 = colfilt(lapResp(:,:,l), [5 5], 'sliding', colfunc1);
            maxMat2 = colfilt(lapResp(:,:,l), [5 5], 'sliding', colfunc2);
            scale_space(:,:,l) = lapResp(:,:,l).*...
                (lapResp(:,:,l) == maxMat1 & maxMat1 ~= maxMat2);
            maxMat(:,:,l) = maxMat1;
        end
        toc
    else
        % ordfilt2
        tic
        for l = 1:nScale
            % Ensure local max is unique
            maxMat1 = ordfilt2(lapResp(:,:,l), 25, ones(5,5));
            maxMat2 = ordfilt2(lapResp(:,:,l), 24, ones(5,5));
            scale_space(:,:,l) = lapResp(:,:,l).*...
                (lapResp(:,:,l) == maxMat1 & maxMat1 ~= maxMat2);
            maxMat(:,:,l) = maxMat1;
        end
        toc
    end
    
    % Inter-slice nonmaximum suppresion
    scale_space = reshape(scale_space, h*w, nScale);
    maxMat = reshape(maxMat, h*w, nScale);
    for l = 1:nScale
        nzIdx = find(scale_space(:,l) ~= 0);
        if (l == 1)
            supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l+1));
        elseif (l == nScale)
            supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l-1));
        else
            supIdx = find(scale_space(nzIdx, l) <= maxMat(nzIdx, l-1) |...
                          scale_space(nzIdx, l) <= maxMat(nzIdx, l+1));
        end
        scale_space(nzIdx(supIdx), l) = 0;
    end
    % Thresholding
    scale_space(scale_space < thres) = 0;    
    
    % Additional thresholding using Harris response and
    % affine adaptation
    for l = 1:nScale
        subplot(2, nScale, nScale+l); imagesc(ceil(reshape(scale_space(:,l), h, w)));
    end
    nzIdx = find(scale_space ~= 0);
    [cy, cx, radIdx] = ind2sub([h, w, nScale], nzIdx);
    rad = scale(radIdx)'*sigma*sqrt(2);
    
    Ix = conv2(im{k}, dx, 'same');    % Image derivatives
    Iy = conv2(im{k}, dy, 'same');
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix.*Iy;
        
    cyOut = -1*ones(numel(nzIdx), 1);
    cxOut = -1*ones(numel(nzIdx), 1);
    radOut = -1*ones(numel(nzIdx), 1);
    lamOut = cell(numel(nzIdx), 1);
    nzIdxOut = -1*ones(numel(nzIdx), 1);
    for l = 1:numel(nzIdx)
        % Use a Gaussian window that is a factor of 1.5 or 2 larger than the characteristic scale of the blob
        g = fspecial('gaussian', floor_odd(6*rad(l)*2/sqrt(2)), rad(l)*2/sqrt(2));
        M = [imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ix2, g) imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ixy, g);...
             imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Ixy, g) imfilterat(cx(l), cy(l), (size(g,1)-1)/2, Iy2, g)];
        R = det(M) - alpha*trace(M)^2; % Harris corner response
        if (R > 0)
            cxOut(l) = cx(l);
            cyOut(l) = cy(l);
            radOut(l) = rad(l);
            [V, D] = eig(M);
            [D, idx] = sort(diag(D)); % Sort eigvec according to eigval
            lamOut{l} = [diag(D), V(:,idx)];
            nzIdxOut(l) = nzIdx(l);
        end
    end
    cyOut(cyOut == -1) = [];
    cxOut(cxOut == -1) = [];
    radOut(radOut == -1) = [];
    lamOut(cellfun('isempty',lamOut)) = [];
    nzIdxOut(nzIdxOut == -1) = [];
    
    % Compute ellipse's param
    a = zeros(numel(lamOut), 1);
    b = zeros(numel(lamOut), 1);
    phi = zeros(numel(lamOut), 1);
    for l = 1:numel(lamOut)
        % Relative shape
        a(l) = lamOut{l}(1,1)^(-1/2); % major axis
        b(l) = lamOut{l}(2,2)^(-1/2); % minor axis
        % Ensure no ambiguity between angle and orientation
        if lamOut{l}(2,3) < 0
            phi(l) = acos(dot(lamOut{l}(:,3),[1; 0])/norm(lamOut{l}(:,3))/norm([1; 0]));
        else
            phi(l) = acos(dot(-lamOut{l}(:,3),[1; 0])/norm(lamOut{l}(:,3))/norm([1; 0]));
        end
        % Absolute scale 
        tmpa = a(l)/(a(l)+b(l))*radOut(l)*2; 
        tmpb = b(l)/(a(l)+b(l))*radOut(l)*2;
        a(l)= tmpa;
        b(l)= tmpb;
    end
    
    % Output results
    disp(numel(rad))
    disp(numel(radOut))
    figure(fgOut); 
    subplot(121);show_all_circles(im{k}, cxOut, cyOut, radOut);
    text(cxOut, cyOut, num2str([1:numel(radOut)]'), 'color', 'y'); % Mark points with indices
    subplot(122);show_all_ellipses(im{k}, cxOut, cyOut, a, b, phi);
    text(cxOut, cyOut, num2str([1:numel(radOut)]'), 'color', 'y');
    set(fgOut, 'units','normalized', 'position', [0 0 1 1])
end

%% Test subroutine
[cxOut, cyOut, radOut, a, b, phi] = blobDetectAffine(im{4}, sigma, scale, thres, alpha);
fgTmp = figure;
subplot(121);show_all_circles(im{k}, cxOut, cyOut, radOut);
text(cxOut, cyOut, num2str([1:numel(radOut)]'), 'color', 'y'); % Mark points with indices
subplot(122);show_all_ellipses(im{k}, cxOut, cyOut, a, b, phi);
text(cxOut, cyOut, num2str([1:numel(radOut)]'), 'color', 'y');
set(fgTmp, 'units','normalized', 'position', [0 0 1 1])