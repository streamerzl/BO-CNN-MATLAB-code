function layers = createCNN(numConvLayers, filterSize, numFilters, dropoutRate, numOutputs)
    inputSize = [36, 27]; % This is dimensions of the input 2-D matrix

    layers = [
        imageInputLayer([36 27 3], 'Normalization', 'none')
    ];

    for i = 1:numConvLayers
        layers = [
            layers
            convolution2dLayer(filterSize, numFilters, 'Stride', 1, 'Padding', 0)
            batchNormalizationLayer
            reluLayer            
        ];
    
        % Add pooling layer and check dimensions 
        layers = [
                    layers
                    maxPooling2dLayer(2, 'Stride', 2)
        ];
        
        inputSize = floor((inputSize - filterSize + 1) ./ 2); % Update input size
        if any(inputSize < 1)
            error('Invalid configuration: Input size too small after pooling.');
        end

    end   
    

    layers = [
        layers
        dropoutLayer(dropoutRate)
        fullyConnectedLayer(numOutputs)
        regressionLayer
    ];
end