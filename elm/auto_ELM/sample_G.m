
nclass=max(fae(:,1));
fdatabase.label=fae(:,1);
tr_idx=[]; ts_idx=[];
for jj = 1:nclass,
        idx_label = find(fdatabase.label == jj);
        num = length(idx_label);
        
        idx_rand = randperm(num);
        tr_num=5;
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
end
    tr_fea=fae(tr_idx,2:end);
    tr_label=fae(tr_idx,1);
    ts_fea=fae(ts_idx,2:end);
    ts_label=fae(ts_idx,1);
    