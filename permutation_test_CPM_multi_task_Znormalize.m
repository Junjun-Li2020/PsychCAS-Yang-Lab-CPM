function [pval,prediction_r]=permutation_test_CPM_multi_task_Znormalize(all_mats,corr_behav,all_behav,all_covariate,thresh,lambda,alpha,no_iterations)

no_sub = size(all_mats,3);

% calculate the true prediction correlation
[leave_one_out,Routput,MSE]= prior_predict_behavior_CPM_multi_task_Znormalize(all_mats,corr_behav,all_behav,all_covariate,thresh,lambda,alpha);
true_prediction_r_pos=Routput(1);
true_prediction_r_neg=Routput(2);
true_prediction_r_bilateral=Routput(3);

% number of iteration for permutation testing
prediction_r = zeros (no_iterations,3);
prediction_r(1,1) = true_prediction_r_pos;
prediction_r(1,2) = true_prediction_r_neg;
prediction_r(1,3) = true_prediction_r_bilateral;

% create estimate distribution of the test statistic
% via random shuffles of data lables
for it=2:no_iterations
    fprintf('\n Performing iteration %d out of %d', it, no_iterations);
    new_behav = all_behav(randperm(no_sub),:);
    [leave_one_out,Routput,MSE]= prior_predict_behavior_CPM_multi_task_Znormalize(all_mats,new_behav,new_behav,all_covariate,thresh,lambda,alpha);
    prediction_r(it,1)=Routput(1);
    prediction_r(it,2)=Routput(2);
    prediction_r(it,3)=Routput(3);
end

sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
position_pos = find(sorted_prediction_r_pos==true_prediction_r_pos);
pval_pos = position_pos(1)/no_iterations;

sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
position_neg = find(sorted_prediction_r_neg==true_prediction_r_neg);
pval_neg = position_neg(1)/no_iterations;

sorted_prediction_r_bilateral = sort(prediction_r(:,3),'descend');
position_bilateral = find(sorted_prediction_r_bilateral==true_prediction_r_bilateral);
pval_bilateral = position_bilateral(1)/no_iterations;

pval=[pval_pos,pval_neg,pval_bilateral];