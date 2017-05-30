
load 'EEG.mat'
X_train_raw = X_Train{1,1};
% 1x45x2010x310
Y_train_raw = Y_Train{1,1};
% 1x45x2010x1
X_test_raw = X_Test{1,1};
%1x45x1384x310
Y_test_raw = Y_Test{1,1};
%1x45x1384x1

%append all data
for j = 2:45
    X_train_raw = [X_train_raw; X_Train{1,j}];
    X_test_raw = [X_test_raw; X_Test{1,j}];
    Y_train_raw = [Y_train_raw; Y_Train{1,j}];
    Y_test_raw = [Y_test_raw; Y_Test{1,j}];
end
size(X_train_raw)
size(Y_train_raw)
X_train = mapminmax(X_train_raw,0, 1);
X_test = mapminmax(X_test_raw, 0, 1);
%data=[faces'];
% transpose
% 400*(64*64)

ori_Training=[Y_train_raw X_train];  % tr_label represents training label, tr_fea represents training features
ori_Testing=[Y_test_raw X_test];  % the same as above

C1=2^2; % you can set as 2^{-20} to 2^{20}
target_dimension=150;  % the desired dimension what you want
loop=2;  %  number of loop in each learning step, it could eqaul 2 to 10, but not too many differences. 

[auto_train_time,auto_test_accuracy,selec_Training,selec_Testing] = auto_ELM(ori_Training,ori_Testing,1,target_dimension ,'sig',C1,loop,1); %% The dimension reduction method comes from my newest Trans Part A publication

size(selec_Training)
size(selec_Testing)
save eeg_feature_150 selec_Training selec_Testing