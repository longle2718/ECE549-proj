function stat = tfidf(d, D)
% Compute tfidf statistics
% Input:
%   d - query document n x numTerm
%   D - document database N x numTerm

n = size(d, 1); % Number of documents in the query set
M = size(d, 2); % Number of terms
N = size(D, 1); % Number of documents in the database

stat = zeros(n, M);
for k = 1:n
    stat(k,:) = d(k,:)/sum(d(k,:)).*log(N./(1+sum(D>0)));
end