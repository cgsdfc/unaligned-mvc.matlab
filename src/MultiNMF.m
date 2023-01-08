function [X, v2_gnd, U, V, centroidV, log, ac, accL] = MultiNMF(X, U, V, centroidV, K, label, options)
    % This is a module of Multi-View Non-negative Matrix Factorization(MultiNMF)
    %
    % Notation:
    % X ... a cell containing all views for the data
    % K ... number of hidden factors
    % label ... ground truth labels
    % Written by Jialu Liu (jliu64@illinois.edu)

    viewNum = length(X);
    Rounds = options.rounds;

    if options.iter == 1
        % if this is the first iteration, initialize U and V and V_centroid.
        % For the rest of time, use the U,V,V_centroid from previous iteration.
        U_ = [];
        V_ = [];

        U = cell(1, viewNum);
        V = cell(1, viewNum);

        j = 0;

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
                [U{i}, V{i}] = NMF(X{i}, K, options, U_, V{i - 1});
                printResult(V{i}, label, K, options.kmeans);
            end

        end

        centroidV = V{1};
    end % end initialization.

    log = [];
    ac = [];
    optionsForPerViewNMF = options;
    oldL = 100;

    %% aligning the initial matrice %%
    align_num = options.align_sum;
    idx_rand = options.idx_rand;
    ugnd = label(align_num + 1:end);

    for i = 2:viewNum

        if options.iter == 1
            v2_gnd{i} = ugnd(idx_rand{i});
        else
            v2_gnd{i} = options.v2_gnd{i};
        end

    end

    % Alignment
    for i = 2:viewNum
        [Perm{i}, v2_gnd{i}, accL{i}] = PermLearn(V{1}(align_num + 1:end, :), V{i}(align_num + 1:end, :), ugnd, v2_gnd{i}, options.t);
        V{i}(align_num + 1:end, :) = Perm{i} * V{i}(align_num + 1:end, :);
        X{i}(:, align_num + 1:end) = (Perm{i} * X{i}(:, align_num + 1:end)')';
        fprintf('the iteration %d, view %d, accL %f\n', options.iter, i, accL{i});
        % disp(['the iteration ', num2str(options.iter), ' view : ', num2str(i), ' accL :', num2str(accL{i})])
    end

    %%
    tic
    j = 0; P_ind = 1;

    while j < Rounds
        j = j + 1;

        % Update centroid.
        centroidV = options.alpha(1) * V{1};

        for i = 2:viewNum
            centroidV = centroidV + options.alpha(i) * V{i};
        end

        centroidV = centroidV / sum(options.alpha);

        % Compute loss.
        logL = 0;

        for i = 1:viewNum
            tmp1 = X{i} - U{i} * V{i}';
            tmp2 = V{i} - centroidV;
            logL = logL + sum(sum(tmp1.^2)) + options.alpha(i) * sum(sum(tmp2.^2));
        end

        % Record history of loss.
        log(end + 1) = logL;
        % logL;

        % if (oldL < logL)
        %     U = oldU;
        %     V = oldV;
        %     logL = oldL;
        %     j = j - 1;
        %     disp('restrart this iteration');
        % else
        %     ac(end + 1) = printResult(centroidV, label, K, options.kmeans);
        % end

        % Test performance of V_centroid.
        ac(end + 1) = printResult(centroidV, label, K, options.kmeans);
        % oldU = U;
        % oldV = V;
        % oldL = logL;

        % Update U,V.
        for i = 1:viewNum
            optionsForPerViewNMF.alpha = options.alpha(i);
            optionsForPerViewNMF.align_sum = options.align_sum;
            [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, optionsForPerViewNMF, U{i}, V{i});
        end

        %% Permute matrix learning for data aligning, Modified %%

        %       if (( mod(j,5)==0 && P_ind ) && options.Per)
        % %       if ( P_ind )
        %        align_sum = options.align_sum;
        %        idx_rand  = options.idx_rand;
        %        ugnd = label(align_sum+1:end);
        %        [Perm2,accL2] = PermLearn(V{1}(align_sum+1:end,:),V{2}(align_sum+1:end,:),ugnd,ugnd(idx_rand{2}));
        %        [Perm3,accL3] = PermLearn(V{1}(align_sum+1:end,:),V{3}(align_sum+1:end,:),ugnd,ugnd(idx_rand{3}));
        %        if ((accL2>0.5) && (accL3>0.5))
        %            P_ind = 0;
        %        end
        %        V{2}(align_sum+1:end,:) = Perm2 * V{2}(align_sum+1:end,:);
        %        V{3}(align_sum+1:end,:) = Perm3 * V{3}(align_sum+1:end,:);
        %        X{2}(:,align_sum+1:end) = (Perm2 * X{2}(:,align_sum+1:end)')';
        %        X{3}(:,align_sum+1:end) = (Perm3 * X{3}(:,align_sum+1:end)')';
        %       end
    end

    toc
