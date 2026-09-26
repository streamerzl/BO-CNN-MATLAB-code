function returnRes=CNNModelWithOptimization(X_totaldata,Y_totaldata,maxObj)   
    maxEpoches=15000;
    %% Step 1: Define Hyperparameter Search Space
    optimVars = [
        optimizableVariable('numFilters', [20, 120], 'Type', 'integer')
        optimizableVariable('filterSize', [2, 10], 'Type', 'integer')
        optimizableVariable('dropoutRate', [0.1, 0.7])
        optimizableVariable('learningRate', [1e-5, 1e-2], 'Transform', 'log')
        optimizableVariable('batchSize', {'32','64', '100', '200','300','400','500'}, 'Type', 'categorical')

    ];
    %% Step 2: Define Objective Function    
    objFcn = @(params)evaluateCNNWithCrossValidation(X_totaldata, Y_totaldata,params,maxEpoches);
    %% Step 3: Perform Bayesian Optimization
    results = bayesopt(objFcn, optimVars, ...
        'MaxObjectiveEvaluations', maxObj, ...        
        'AcquisitionFunctionName', 'expected-improvement-plus');
    %% Step 4: Save Best Parameters
    bestParams = results.XAtMinObjective;
    save('bestParams.mat', 'bestParams');
    bestMetric = results.MinObjective;
	returnRes=bestMetric;
end