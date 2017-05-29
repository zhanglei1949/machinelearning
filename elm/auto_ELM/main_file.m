
    load olivettifaces.mat
label = [];
for j = 1 : 40
label = [label; j * ones(10,1)];
end;

data=[faces'];
data=mapminmax(data,0,1); 
    fae=[label data];

        sample_G;
    clear fae
Training=[tr_label tr_fea];  % tr_label represents training label, tr_fea represents training features
Testing=[ts_label ts_fea];  % the same as above



C1=2^2; % you can set as 2^{-20} to 2^{20}
target_dimension=200;  % the desired dimension what you want
loop=2;  %  number of loop in each learning step, it could eqaul 2 to 10, but not too many differences. 



[train_time,test_accuracy2,Training,Testing]=auto_ELM(Training,Testing,1,target_dimension,'sine',C1,loop,0); %% The dimension reduction method comes from my newest Trans Part A publication

number_subnetwork_node=1;

C2=2^2;  % you can set as 2^{-20} to 2^{20}

[train_time,  train_accuracy,test_accuracy]=MFB_ELM(Training,Testing,1,1,'sig',number_subnetwork_node,C2);    

%%% The method comes from my paper:  Yang Yimin, Q.M.Jonathan Wu. ¡°Extreme Learning Machine with Subnetwork Hidden Nodes for Regression and Classification Problems.¡± 
                                                                                            %IEEE Transactions on Cybernetics. In press.  
                                                                                            % if you are interesting, I can provide you the m file. Here I just give you the P file as I need some time to find my m file. 


[train_time,  train_accuracy,test_accuracy1]=ELM(Training,Testing,1,1000,'sig',2^-1);    %%% Original ELM