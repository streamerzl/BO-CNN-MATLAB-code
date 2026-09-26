function valRMSE = evaluateCNNWithCrossValidation(X_totaldata, Y_totaldata, params,maxEpochs)
    % Cross-validation logic
    numFolds = 5;
    indices = crossvalind('Kfold', size(Y_totaldata, 1), numFolds);
    rmseValues = zeros(numFolds, 1);

    for fold = 1:numFolds
        valIdx = (indices == fold);
        trainIdx = ~valIdx;

        XTrainFold = X_totaldata(:, :, :, trainIdx);
        YTrainFold = Y_totaldata(trainIdx, :);
        XValFold = X_totaldata(:, :, :, valIdx);
        YValFold = Y_totaldata(valIdx, :);

        % Try to create and validate layers
        try
            layers = createCNN(1, params.filterSize, params.numFilters, params.dropoutRate, size(Y_totaldata, 2));
        catch ME
            % If invalid, return a high RMSE to skip this configuration
            disp(['Skipping invalid configuration: ', ME.message]);
            valRMSE = Inf;
            return;
        end         
       
        options = trainingOptions('adam', ...
            'InitialLearnRate', params.learningRate, ...
            'MaxEpochs', maxEpochs, ...
            'MiniBatchSize', str2double(string(params.batchSize)), ...
            'Shuffle', 'every-epoch', ...            
            'ExecutionEnvironment', 'multi-gpu', ...
            'Verbose', false);

        net = trainNetwork(XTrainFold, YTrainFold, layers, options);
        YPred = predict(net, XValFold);
        rmseValues(fold) = sqrt(mean((YValFold - YPred).^2, 'all'));
    end

    valRMSE = mean(rmseValues);
end