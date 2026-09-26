function MainCNN
    clc;
    clear;
    num_repetition=10;%Variabel
    maxObj=30;%Variabel
    path='F:\ResearchTopic\CNN\FinalCNNCodeforFWE\BOCNN\SingleLayer\FWE4';
    RMSERes=zeros(num_repetition,length(maxObj));
    
    for j=1:1
        for i=1:num_repetition
            load(['X_data',num2str(i),'.mat'],'X_totaldata');
            load(['Y_data',num2str(i),'.mat'],'Y_totaldata');
            RMSERes(i,j)=CNNModelWithOptimization(X_totaldata,Y_totaldata,maxObj(j));
            path_old=path;        
            path_new=[path_old,'\results'];        
            movefile([path_old,'\bestParams.mat'],[path_new,'\bestParams_',num2str(j),'_',num2str(i),'.mat']);
            save('RMSERes.txt','RMSERes','-ascii');
        end
    end
end