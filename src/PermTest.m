% MIT License
% 
% Copyright (c) 2022 Ao Li, Cong Feng
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

clear;
%% parameter setting
options = [];
options.maxIter = 200;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 30;


% options.kmeans means whether to run kmeans on v^* or not
% options alpha is an array of weights for different views

options.alpha = [0.01 0.01 0.01]; %weights for views
options.kmeans = 1;


%% read dataset
addpath(genpath(pwd));
% load ../handwritten.mat % load external .mat file
load ORL_mtv.mat; data = X; gnd = gt;
% load handwritten.mat;data{1} = fourier';data{2} = pixel';
K = length(unique(gnd)); % Cluster number


%% normalize data matrix
% for v = 1:length(data);
%     data{v} = DataNormal(data{v});
% end
for i = 1:length(data)
    data{i} = data{i} / sum(sum(data{i}));
end
%% Simulated Unaligned %%
ratio = 0.70;
[alignX,adjustX,align_gnd,unalign_gnd,align_sum] = unaligned(data,gnd,ratio);
idx_rand = cell(1,length(data)-1);
unalignX = cell(1,length(data));
%% Unaligned Operation for view 2 to end %%
% With this method, we take the first view as the
% baseline and fix it, which means that the truth 
% lables(gnd_new) are corresponding to first view .
unalignX{1} = adjustX{1};
for v = 2:length(data)
    uda = adjustX{v};
    idx_rand{v} = randperm(size(uda,2));
    unalignX{v} = uda(:,idx_rand{v});
end
clear uda;
options.align_sum = align_sum;
options.idx_rand = idx_rand;
aX = cell(1,length(data));
X = cell(1,length(data));
%% Unaligned Data Generation %%
for v = 1:length(data)
    aX{v} = [alignX{v} adjustX{v}];
    X{v}  = [alignX{v} unalignX{v}];
end
gnd_new = [align_gnd;unalign_gnd];

% run 3 times
U_final = cell(1,3);
V_final = cell(1,3);
V_centroid = cell(1,3);
options.Per = 1;
label = gnd_new;

%% Multi NMF
viewNum = length(X);
Rounds = options.rounds;

U_ = [];
V_ = [];

U = cell(1, viewNum);
V = cell(1, viewNum);

j = 0;
log = 0;
ac = 0;

% initialize basis and coefficient matrices
while j < 3
    j = j + 1;
    if j == 1
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V_);
        printResult(V{1}, label, K, options.kmeans);
    else
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V{viewNum});
        printResult(V{1}, label, K, options.kmeans);        
    end
    for i = 2:viewNum
        [U{i}, V{i}] = NMF(X{i}, K, options, U_, V{i-1});
        printResult(V{i}, label, K, options.kmeans);
    end
end

optionsForPerViewNMF = options;
oldL = 100;

%% aligning the initial matrice %%
       align_num = options.align_sum;
       idx_rand  = options.idx_rand;
       ugnd = label(align_num+1:end);
       [Perm2,accL2] = PermLearn(V{1}(align_num+1:end,:),V{2}(align_num+1:end,:),ugnd,ugnd(idx_rand{2}));
       [Perm3,accL3] = PermLearn(V{1}(align_num+1:end,:),V{3}(align_num+1:end,:),ugnd,ugnd(idx_rand{3}));
       V{2}(align_num+1:end,:) = Perm2 * V{2}(align_num+1:end,:);
       V{3}(align_num+1:end,:) = Perm3 * V{3}(align_num+1:end,:);
       X{2}(:,align_num+1:end) = (Perm2 * X{2}(:,align_num+1:end)')';
       X{3}(:,align_num+1:end) = (Perm3 * X{3}(:,align_num+1:end)')';
