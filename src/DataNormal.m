function XX = DataNormal(X)
%% Adaptive to Yale and ORL %%
XX = X./(repmat(sqrt(sum(X.^2,1)),size(X,1),1)+10e-10);

%% Adaptive to UCI %%
% for  j = 1:size(X,2)
%      XX(:,j) = ( X(:,j) - mean( X(:,j) ) ) / std( X(:,j) ) ;
% end
