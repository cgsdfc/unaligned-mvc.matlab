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
