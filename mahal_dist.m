function distMat = mahal_dist(D, C, SIGinv)
% Mahalanobis distance
% Input
%   D - dim x M
%   C - dim x N
%   SIGinv - dim x dim
%
% Long Le
% longle1@illinois.edu
% University of Illinois
%

M = size(D, 2);
N = size(C, 2);

distMat = zeros(M, N);
for k = 1:M
    for l = 1:N
        distMat(k, l) = sqrt((D(:,k)-C(:,l))'*SIGinv*(D(:,k)-C(:,l)));
    end
end