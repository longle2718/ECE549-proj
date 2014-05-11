% Google Video demo
%
% Long Le
% longle1@illinois.edu
% University of Illinois
%

clear all; close all

% Load visual database
load demo.mat wFreqVec d f track SIGinv C
nFrame = size(wFreqVec, 1);
K = size(wFreqVec, 2);

% Ask for frame to select
idx = input(sprintf('Please select a frame to search for (among %d frames):', nFrame));
img = imread(sprintf('imTrain/%.3d.jpg',idx));
figure; imshow(img);

disp('Select object by specifying a rectangle')
while 1
    disp(' ')
    disp('Click first point')
    [x,y,b] = ginput(1);    
    if b=='q'        
        break;
    end
    x1 = x; y1 = y;
    
    disp('Click second point');
    [x,y] = ginput(1);
    x2 = x; y2 = y;
    
    % Assertion: x2, y2 > x1, y1
    rectangle('position', [x1 y1 x2-x1 y2-y1], 'EdgeColor', 'y')
end

% Form query weighted frequency vector
sel = f{idx}(1, :) >= x1 & f{idx}(1, :) <= x2 & f{idx}(2, :) >= y1 & f{idx}(2, :) <= y2 ;

imshow(img)
h = vl_plotframe(f{idx}(:,track{idx}~=0 & sel));
set(h,'color','y','linewidth',2);
if (sum(sel & track{idx}~=0) == 0)
    error('No object detected in the selected window!')
end 

distMat = mahal_dist(d{idx}(:,track{idx}~=0 & sel), C, SIGinv);
[~, idx] = min(distMat, [], 2);
cntVecQ = accumarray(idx, 1, [K 1])';

wFreqVecQ = tfidf(cntVecQ, cntVec);

score = wFreqVec*wFreqVecQ'/norm(wFreqVecQ)./sqrt(sum(wFreqVec.^2, 2));
score(isnan(score)) = 0; % Ignore wFreqVec with norm 0, i.e. frames with no or trivial words
[sortScore, frameIdx] = sort(score, 'descend');

figure;
set(gcf, 'units','normalized', 'position', [0 0 1 1])
for k = 1:15
    subplot(3,5,k); imshow(img{frameIdx(k)})
    h = vl_plotframe(f{frameIdx(k)}(:,track{frameIdx(k)}~=0));
    set(h,'color','y','linewidth',2);
    xlabel(sprintf('Relevance: %0.4f, Frame index: %d', sortScore(k), frameIdx(k)))
    %set(get(gca,'YLabel'),'Rotation',0)
end