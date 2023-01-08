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

function [result, his, iter_opt] = AlignMultiNMF(X, gnd, options)
    %% parameter setting

    options.maxIter = 200;
    options.error = 1e-6;
    options.nRepeat = 30;
    options.minIter = 50;
    options.meanFitRatio = 0.1;
    options.rounds = 30;
    options.trainTimes = 100;
    
    K = length(unique(gnd));

    % options.kmeans means whether to run kmeans on v^* or not
    % options alpha is an array of weights for different views

    try
        options.alpha = options.view_weight * ones(1, length(X));
        options.t = options.kernel_width;
    catch
        options.alpha = [0.01 0.01 0.01]; %weights for views
        options.t = 1;
    end

    options.kmeans = 1;

    if options.debug
        options.maxIter = 1;
        options.nRepeat = 1;
        options.minIter = 1;
        options.rounds = 2;
        options.trainTimes = 1;
    end

    for i = 1:length(X)
        X{i} = X{i} / sum(sum(X{i}));
    end

    U = [];
    V = [];
    V_centroid = [];
    options.Per = 1;
    viewNum = length(X);

    his = [];

    acc_opt = 0;

    for i = 1:options.trainTimes
        options.iter = i;
        [X, v2_gnd, U, V, V_centroid, log, acc, accL] = MultiNMF(X, U, V, V_centroid, K, gnd, options);

        for j = 2:viewNum
            options.v2_gnd{j} = v2_gnd{j};
        end

        fprintf('the iteration %d, acc %f, loss %f\n', i, acc(end), log(end));
        
        his = [his accL];

    end

    result = litekmeans(V_centroid, K, 'Replicates', 20);
end
