function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = DSWAN(train_x,train_y,test_x,test_y,b1,b2,b3,s,C,lam,n1)
tic
train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];

NumberofTrainingData=size(train_x,1);
NumberofTestingData=size(test_x,1);
NumberofInputNeurons=size(train_x,2);
T=train_y';
TV.T=test_y';
sorted_target=sort(cat(2,train_y',test_y'),2);
label=zeros(1,1);
label(1,1)=sorted_target(1,1);
j=1;
for i=2:(NumberofTrainingData+NumberofTestingData)
    if sorted_target(1,i)~=label(1,j)
        j=j+1;
        label(1,j)=sorted_target(1,i);
    end
end
number_class=j;
NumberofOutputNeurons=number_class;

temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break; 
        end
    end
    temp_T(j,i)=1;
end
T=temp_T*2-1;

temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TV.T(1,i)
            break; 
        end
    end
    temp_TV_T(j,i)=1;
end
TV.T=temp_TV_T*2-1;
%% First layer RELM
clear b1;
beta1 = elm_autoencoder(H1,n1);
clear A1;

T1 = H1 * beta1;

fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));

[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';

clear H1;
%% Second layer RELM
H2 = [T1 .1 * ones(size(T1,1),1)];
clear T1;
clear b2;
beta2 = elm_autoencoder(H2,n1);
clear A2;

T2 = H2 * beta2;
fprintf(1,'Layer 2: Max Val of Output %f Min Val %f\n',max(T2(:)),min(T2(:)));

[T2,ps2] = mapminmax(T2',0,1);T2 = T2';

clear H2;
%% Original ELM
H3 = [T2 .1 * ones(size(T2,1),1)];
clear T2;
T3 = H3 * b3;
l3 = max(max(T3));
l3 = s/l3;
fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',l3,min(T3(:)));

T3 = tansig(T3 * l3);
clear H3;
%% Finsh Training
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  T');
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%% Calculate the training accuracy
xx = T3 * beta;
clear T3;

yy = result_tra(xx);
train_yy = result_tra(T');
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
%% First layer feedforward
tic;

test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
clear test_x;

TT1 = HH1 * beta1;TT1  =  mapminmax('apply',TT1',ps1)';
clear HH1;clear beta1;
%% Second layer feedforward
HH2 = [TT1 .1 * ones(size(TT1,1),1)];
clear TT1;

TT2  =  HH2 * beta2;TT2  =  mapminmax('apply',TT2',ps2)';
clear HH2;clear beta2;
%% Last layer feedforward
HH3 = [TT2 .1 * ones(size(TT2,1),1)];
clear TT2;

TT3 = tansig(HH3 * b3 * l3);
clear HH3;clear b3;

x = TT3 * beta;
y = result_tra(x);
test_yy = result_tra(TV.T');
TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
clear TT3;
%% Calculate the testing accuracy
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
