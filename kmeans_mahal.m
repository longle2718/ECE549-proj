function [C, A, minsumd] = kmeans_mahal(D, K, SIGinv, NumRepetitions, maxIter)
%
% K-means using Mahalanobis distance
% Output
%   C - dim x K
%   A - 1 x numObservations
%

Citer = cell(1, NumRepetitions);
Aiter = cell(1, NumRepetitions);
sumd = zeros(1, NumRepetitions);
for k = 1:NumRepetitions
    % Initialization
    Ccurr = D(:, randsample(size(D, 2), K));
    for iter = 1:maxIter
        % Compute the distance matrix
        distMat = mahal_dist(D, Ccurr, SIGinv);
        
        % Find the new assignment
        [dist, Anew] = min(distMat, [], 2);
        
        % Compute new centroids
        Cnew = zeros(size(Ccurr));
        for l = 1:K
            Cnew(:,l) = mean(D(:, Anew == l), 2);
        end
        
        % Check for convergence
        disp(norm(Ccurr - Cnew))
        if norm(Ccurr - Cnew) < 1e-3
            break;
        end
        Ccurr = Cnew;
    end
    Citer{k} = Cnew;
    Aiter{k} = Anew;
    sumd(k) = sum(dist)
end

[minsumd, idx]= min(sumd);
C = Citer{idx};
A = Aiter{idx}';

end