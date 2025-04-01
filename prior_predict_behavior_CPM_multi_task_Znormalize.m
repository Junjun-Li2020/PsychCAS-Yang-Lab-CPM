function [leave_one_out,Routput,MSE]= prior_predict_behavior_CPM_multi_task_Znormalize(all_mats,corr_behav,all_behav,all_covariate,thresh,lambda,alpha)
no_sub = size(all_mats,3);
no_node = size(all_mats,1);
no_task = size(all_mats,4);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred_bilateral = zeros(no_sub,1);

Binary_edge_pos=zeros(no_node*no_node*no_task,no_sub);
Binary_edge_neg=zeros(no_node*no_node*no_task,no_sub);
Binary_edge_bilateral=zeros(no_node*no_node*no_task,no_sub);

num_pos_edges=zeros(no_sub,no_task);  num_neg_edges=zeros(no_sub,no_task);

for i=1:no_node
    for j=i:no_node
        all_mats(i,j,:,:)=0;
    end
end
%% Internal validation: prediction from task connectivity
for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    train_mats = all_mats;
    train_mats(:,:,leftout,:) = [];
    train_vcts_task = reshape(train_mats,[],size(train_mats,3),no_task);
    train_vcts=[];
    for j_task=1:no_task
        train_vcts = [train_vcts; train_vcts_task(:,:,j_task)]; 
    end
    
    t_train_corr_behav=corr_behav;
    t_train_corr_behav(leftout,:) = [];
    train_corr_behav = sum(nanzscore(t_train_corr_behav),2);
    
    t_train_behav = all_behav;
    t_train_behav(leftout,:) = [];
    train_behav = sum(nanzscore(t_train_behav),2);
    
    train_covariate=all_covariate;
    train_covariate(leftout,:) = [];
    
    mean_train_behav = nanmean(t_train_behav);
    std_train_behav = nanstd(t_train_behav);
    transform_test_behav(leftout,1) = sum((all_behav(leftout,:) - mean_train_behav)./std_train_behav);
    
    % correlate all edges with behavior 
    [r_mat,p_mat] = partialcorr(train_vcts',train_corr_behav,train_covariate,'type','Pearson');
    
    
    % set threshold and define masks
    pos_mask = zeros(no_node*no_node*no_task,1);
    neg_mask = zeros(no_node*no_node*no_task,1);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    for j_task=1:no_task 
        a=(no_node*no_node)*(j_task-1)+1;
        b=(no_node*no_node)*j_task;
        num_pos_edges(leftout,j_task)=sum(pos_mask(a:b,1));
        num_neg_edges(leftout,j_task)=sum(neg_mask(a:b,1));
    end
    
    train_sumpos = zeros(no_sub-1,no_task);
    train_sumneg = zeros(no_sub-1,no_task);
    
    for ss = 1:size(train_sumpos);
        for j_task=1:no_task
            a=(no_node*no_node)*(j_task-1)+1;
            b=(no_node*no_node)*j_task;
            train_sumpos(ss,j_task) = sum(train_vcts(a:b,ss).*pos_mask(a:b,1)); %每个被试每个任务显著正相关的边的连接强度之和
            train_sumneg(ss,j_task) = sum(train_vcts(a:b,ss).*neg_mask(a:b,1));
        end
    end
    train_sumbilateral=[train_sumpos,train_sumneg];
    
    AllIndv_pos=[train_sumpos];
    AllIndv_neg=[train_sumneg];
    AllIndv_bilateral=[train_sumpos, train_sumneg];
    
    fit_coef_pos=[]; fit_info_pos=[];
    % build model on TRAIN subs
    if ~exist('lambda', 'var')  || length(lambda) ~= 1
        [fit_coef_pos, fit_info_pos] = lasso(AllIndv_pos, train_behav, 'Alpha',alpha,'CV',no_sub-1,'NumLambda', 100,'LambdaRatio', 1e-10);
        idxLambda1SE_pos = fit_info_pos.Index1SE;
        coef_pos = fit_coef_pos(:,idxLambda1SE_pos);
        coef0_pos = fit_info_pos.Intercept(idxLambda1SE_pos);
        lambda_total_pos(leftout) = fit_info_pos.Lambda(idxLambda1SE_pos);
    else
        [coef_pos, fit_info_pos] = lasso(AllIndv_pos, train_behav, 'Alpha',alpha, 'Lambda', lambda); %lambda=0时结果等同于一般的回归分析
        coef0_pos = fit_info_pos.Intercept;
    end
    
    fit_coef_neg=[]; fit_info_neg=[];
    if ~exist('lambda', 'var')  || length(lambda) ~= 1
        [fit_coef_neg, fit_info_neg] = lasso(AllIndv_neg, train_behav, 'Alpha',alpha,'CV',no_sub-1,'NumLambda', 100,'LambdaRatio', 1e-10);
        idxLambda1SE_neg = fit_info_neg.Index1SE;
        coef_neg = fit_coef_neg(:,idxLambda1SE_neg);
        coef0_neg = fit_info_neg.Intercept(idxLambda1SE_neg);
        lambda_total_neg(leftout) = fit_info_neg.Lambda(idxLambda1SE_neg);
    else
        [coef_neg, fit_info_neg] = lasso(AllIndv_neg, train_behav, 'Alpha',alpha, 'Lambda', lambda);
        coef0_neg = fit_info_neg.Intercept;
    end
    
    fit_coef_bilateral=[]; fit_info_bilateral=[];
    if ~exist('lambda', 'var')  || length(lambda) ~= 1
        [fit_coef_bilateral, fit_info_bilateral] = lasso(AllIndv_bilateral, train_behav, 'Alpha',alpha,'CV',no_sub-1,'NumLambda', 100,'LambdaRatio', 1e-10);
        idxLambda1SE_bilateral = fit_info_bilateral.Index1SE;
        coef_bilateral = fit_coef_bilateral(:,idxLambda1SE_bilateral);
        coef0_bilateral = fit_info_bilateral.Intercept(idxLambda1SE_bilateral);
        lambda_total_bilateral(leftout) = fit_info_bilateral.Lambda(idxLambda1SE_bilateral);
    else
        [coef_bilateral, fit_info_bilateral] = lasso(AllIndv_bilateral, train_behav, 'Alpha',alpha, 'Lambda', lambda);
        coef0_bilateral = fit_info_bilateral.Intercept;
    end
    
    % edges for prediction
    Binary_edge_pos(pos_edges,leftout)=1;
    Binary_edge_neg(neg_edges,leftout)=1;
    Binary_edge_bilateral([pos_edges;neg_edges],leftout)=1;
    
    % run model on TEST sub
    test_mat = squeeze(all_mats(:,:,leftout,:));
    test_vct_task = reshape(test_mat,[],no_task);
    test_vct=[];
    for j_task=1:no_task
        test_vct = [test_vct; test_vct_task(:,j_task)];
    end
    
    for j_task=1:no_task
        a=(no_node*no_node)*(j_task-1)+1;
        b=(no_node*no_node)*j_task;
        test_sumpos(1,j_task) =nansum(test_vct(a:b,1).*pos_mask(a:b,1));
        test_sumneg(1,j_task) =nansum(test_vct(a:b,1).*neg_mask(a:b,1));
    end
    test_sumbilateral=[test_sumpos,test_sumneg];

    behav_pred_pos(leftout) = test_sumpos*coef_pos + coef0_pos;
    behav_pred_neg(leftout) = test_sumneg*coef_neg + coef0_neg;
    behav_pred_bilateral(leftout) = test_sumbilateral*coef_bilateral + coef0_bilateral; 
end

leave_one_out.num_edges.pos=num_pos_edges; leave_one_out.num_edges.neg=num_neg_edges; 
leave_one_out.behav_pred = [behav_pred_pos,behav_pred_neg,behav_pred_bilateral];
leave_one_out.transform_test_behav = transform_test_behav;

% compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,transform_test_behav);
[R_neg, P_neg] = corr(behav_pred_neg,transform_test_behav);
[R_bilateral, P_bilateral] = corr(behav_pred_bilateral,transform_test_behav);
Routput=[R_pos,R_neg,R_bilateral];
leave_one_out.R=[R_pos,R_neg,R_bilateral]; leave_one_out.P_parameter_test=[P_pos,P_neg,P_bilateral];

% compare predicted and observed scores using mean squared error
MSE_pos = sum((behav_pred_pos-transform_test_behav).^2)/(no_sub-length(coef_pos));
MSE_neg = sum((behav_pred_neg-transform_test_behav).^2)/(no_sub-length(coef_neg));
MSE_bilateral = sum((behav_pred_bilateral-transform_test_behav).^2)/(no_sub-length(coef_bilateral));
MSE=[MSE_pos,MSE_neg,MSE_bilateral];
leave_one_out.MSE=[MSE_pos,MSE_neg,MSE_bilateral];

%Identifying FC used for prediction
Binary_edge_pos=sum(Binary_edge_pos,2);
Binary_edge_neg=sum(Binary_edge_neg,2);
Binary_edge_bilateral=sum(Binary_edge_bilateral,2);
for j_task=1:no_task
    a=(no_node*no_node)*(j_task-1)+1;
    b=(no_node*no_node)*j_task;
    Binary_mats_pos(:,:,j_task)=reshape(Binary_edge_pos(a:b,1),no_node,no_node);
    Binary_mats_neg(:,:,j_task)=reshape(Binary_edge_neg(a:b,1),no_node,no_node);
    Binary_mats_bilateral(:,:,j_task)=reshape(Binary_edge_bilateral(a:b,1),no_node,no_node);
end

leave_one_out.Binary_edge.pos=Binary_edge_pos;
leave_one_out.Binary_edge.neg=Binary_edge_neg;
leave_one_out.Binary_edge.bilateral=Binary_edge_bilateral;

leave_one_out.Binary_mats.pos=Binary_mats_pos;
leave_one_out.Binary_mats.neg=Binary_mats_neg;
leave_one_out.Binary_mats.bilateral=Binary_mats_bilateral;