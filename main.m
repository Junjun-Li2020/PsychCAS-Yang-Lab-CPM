%% CPM
clear;clc
cd E:\CPM_scripts % script pathway

load('E:\CPM_analysis_data\AD_GFC.mat') % GFC matrix
load('E:\CPM_analysis_data\AD_IndVar.mat'); % Behavior index
load('E:\CPM_analysis_data\AD_Cov.mat'); % Covariates

Cov=[AD_Cov.age, AD_Cov.sex, AD_Cov.FD];

%IndVar =[1./AD_IndVar.CopyTimeHFCfast,1./AD_IndVar.CopyTimeLFCfast]; % handwriting speed
IndVar =[AD_IndVar.ChineseReading]; % reading score
delect_subnum=[18 20 23 29 46 50]'; % Delete subjects with excessive head movements
indexNaN=[];
for j=1:size(IndVar,2)
    EachIndVar_indexNaN = find(isnan(IndVar(:,j))==1);
    indexNaN = [indexNaN; EachIndVar_indexNaN];
end
delect_subnum = [delect_subnum; indexNaN];

all_mats=[];
all_mats1= AD_GFC; all_mats1(:,:,delect_subnum,:)=[]; all_mats(:,:,:,1)=all_mats1;

corr_behav=IndVar; corr_behav(delect_subnum,:)=[];
all_behav=IndVar; all_behav(delect_subnum,:)=[];

all_covariate=Cov; all_covariate(delect_subnum,:)=[];

threshold=[0.0005 0.001 0.0025]; lambda=[]; alpha=1; num_thresh=0.9;
Result_CPM.threshold=threshold;
for i=1:length(threshold)
    thresh=threshold(i)
    [leave_one_out,Routput,MSE]= prior_predict_behavior_CPM_multi_task_Znormalize(all_mats,corr_behav,all_behav,all_covariate,thresh,lambda,alpha);
    Result_CPM.leave_one_out{i,1}=leave_one_out;
    Result_CPM.allthresR(i,:)=Routput;
    Result_CPM.allthresMSE(i,:)=MSE;
end

%% permutation
no_iterations=1000;
for i=1:length(threshold)
    thresh=threshold(i)
    [pval,prediction_r]=permutation_test_CPM_multi_task_Znormalize(all_mats,corr_behav,all_behav,all_covariate,thresh,lambda,alpha,no_iterations);
    Result_CPM.allpval(i,:)=pval;
end
