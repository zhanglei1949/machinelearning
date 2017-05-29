function [TrainingTime, TestingAccuracy,Training,Testing] = auto_ELM(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C,kkkk,type)

%
% Input:
% train_data            - training data set
% test_data             - testing data set
% Elm_Type              - 1 for classification only
% NumberofHiddenNeurons - desired dimensional target
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                          
% kkkk                  - Number of internal loop (normally set 2).
% type                  - activation usage. 
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for reconstruction error
%
% 
    %%%%    Authors:    YIMIN YANG, Q.M.Jonathan Wu, and YAONAN WANG
    %%%%    DATE:       APRIL 2017
    %%%%   The current version is still not optimized. 
%%%%%%%%%%% Macro definition
REGRESSION=0;
fdafe=0;
%%%%%%%%%%% Load training dataset

T=train_data(:,1)';
OrgT=T;
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset

TV.T=test_data(:,1)';
OrgTT=TV.T;
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=(temp_T*2-1);

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=(temp_TV_T*2-1);

end                                                 %   end if of Elm_Type
start_time_train=cputime;
tic;
T=P;
TV.T=TV.P;

%%%%%%%%%%% Calculate weights & biases

for j=1:kkkk
    if j==1
        count=1;
    else
        count=1;
    end
    
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
for nxh=1:count
if j==1
    
    BiasofHiddenNeurons1=rand(NumberofHiddenNeurons,1);
        BiasofHiddenNeurons1=orth(BiasofHiddenNeurons1);
    BBP=BiasofHiddenNeurons1;
InputWeight1=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

        if NumberofHiddenNeurons > NumberofInputNeurons
            InputWeight1 = orth(InputWeight1);
        else
            InputWeight1 = orth(InputWeight1')';
        end

YYM_H=InputWeight1*[P TV.P];
YYM_tempH= bsxfun(@minus, YYM_H', BBP.')';

%%%%%%%%%%% Or directly inherit the previous parameters 

else
        if nxh==1
        clear save_H_YYM
        end
    clear PP
    P=P_save;
    clear H
    
   InputWeight1=YYM;  % inherit the previous parameters
   fdafe=0;
   YYM_H=InputWeight1*[P TV.P];

   clear PP1

BBP=BB; %inherit the previous parameters

YYM_tempH= bsxfun(@minus, YYM_H', BBP.')';
clear YYM_H
end


clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
clear  BiasMatrix 

if type==1
    
    switch lower(ActivationFunction)
case {'sig','sigmoid'}
YYM_H   = 1 ./ (1 + exp(-YYM_tempH));
  case {'sin','sine'}
     YYM_H = sin(YYM_tempH);  
        case  {'radbas'}
              YYM_H = radbas(YYM_tempH);  
                  case {'hardlim'}
        YYM_H = double(hardlim(YYM_tempH));
    end

else
 YYM_H = YYM_tempH;  
end

    clear YYM_tempH

YYM_H=mapminmax(YYM_H,-1,1);
H=YYM_H(:,1:NumberofTrainingData);



end



P_save=P;

P=H;
clear H
FT=zeros(3,17766);
E1=T;



for i=1:1

Y2=E1;

clear tempH


switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        [Y22,PS(1)]=mapminmax(Y2,0.01,0.99);
    case {'sin','sine','radbas','hardlim'}
        %%%%%%%% Sine
       [Y22,PS(1)]=mapminmax(Y2,-1,1);
end




Y2=Y22;
clear Y22

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
Y4=(-log((1./Y2)-1)); 


    case {'sin','sine','radbas','hardlim'}
        %%%%%%%% Sine
       Y4=asin(Y2);
end

Y4=real(Y4);

clear Y22
clear Y2

if fdafe==0

YYM=(eye(size(P,1))/C+P * P') \ P *Y4';


YJX=(P)'*YYM;

BB1=size(Y4);
BB2=sum(YJX-Y4');
clear Y4
BB=BB2/BB1(2);
BB=BB(1);
else
    

YJX=P'*YYM;

end






GXZ111 = bsxfun(@minus, YJX', BB.')';
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ111'));
    case {'sin','sine','radbas','hardlim'}
        %%%%%%%% Sine
GXZ2=sin(GXZ111');
end



FYY = mapminmax('reverse',GXZ2,PS(1))';

clear GXZ2
clear GXZ111
clear Y4
clear YJX


FT1{i}=FYY';


E1=E1-FT1{i};

if i==1
FT=FT1{i};
else
    FT=FT+FT1{i};
end


if i==1
fdafe=1;
end


end

PP=P;
end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;

i=1;
clear YYM_PP
clear layer1st_H
clear PP
clear P_save

Training=[OrgT ;P]';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Testing %%%%%%%%%%%%%%%%%


clear E1
clear GXZ111
clear FYY
clear FT1

H_test = YYM_H(:,NumberofTrainingData+1:end);


clear P


Testing=[OrgTT; H_test]';


GXZ11=H_test'*YYM;
clear H_test
GXZ111= bsxfun(@minus,GXZ11', BB.')';

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ111'));
    case {'sin','sine','radbas','hardlim'}
        %%%%%%%% Sine
GXZ2=sin(GXZ111');
end
clear GXZ111
clear GXZ11
FYY = mapminmax.reverse(GXZ2,PS(i))';
clear GXZ2
FYY=FYY;
TY2=FYY;
clear FYY

TV.T=TV.T;
TY2=TY2';





 TestingAccuracy=sqrt(mse(TY2-TV.T));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%

