clc
clear
close all
%% Preprocessing
DS=readtable('DS_LinkedIn.csv');
DS.authorJob=string(DS.authorJob);
DS.authorJob(DS.authorJob=='company')=3;
DS.authorJob(DS.authorJob=='senior')=2;
DS.authorJob(DS.authorJob=='junior')=1;
DS.authorJob(DS.authorJob=='student')=0;
DS.authorJob=double(DS.authorJob);
DS.followers=double(string(DS.followers));
DS.topic=string(DS.authorJob);
DS.topic(DS.topic=='job')=3;
DS.topic(DS.topic=='technical')=2;
DS.topic(DS.topic=='social')=1;
DS.authorJob(DS.topic=='etc')=0;
DS.topic=double(DS.topic);
DS.vote=DS.vote+0.0001;
head(DS);
tail(DS);
DS_O= isoutlier(DS(:,[2 7]));
idx_O=find(sum(DS_O,2)>=2);
DS(idx_O,:)=[];
IN=table2array(DS(:,1:end-1));
Label=table2array(DS(:,end));

%% Holdout
% trainingData= IN(1:235,:);
% trainingLabels=Label(1:235,:);
% 
% testData=IN(251:end,:);
% testLabels=Label(236:end,:);
%% KFold == 10
KFold=15;

for kfold_index=1:KFold
    index_start=fix((kfold_index-1)*(1/KFold)*size(IN,1))+1;
    index_end=fix((kfold_index)*(1/KFold)*size(IN,1));
    testData_kfold=IN(index_start:index_end,:);
    testLabels_kfold=Label(index_start:index_end,:);
    trainingData_kfold=IN([1:index_start,index_end+1:end],:);
    trainingLabels_kfold=Label([1:index_start,index_end+1:end]);
    %% normalization
for i=[2 7 9]
    MAXX_kfold(i)=max(trainingData_kfold(:,i));
    MINN_kfold(i)=min(trainingData_kfold(:,i));
    trainingData_kfold(:,i)=(trainingData_kfold(:,i)-MINN_kfold(i))/(MAXX_kfold(i)-MINN_kfold(i));
    testData_kfold(:,i)=(testData_kfold(:,i)-MINN_kfold(i))/(MAXX_kfold(i)-MINN_kfold(i));  
end

%% training
Model_kfold= fitcnb(trainingData_kfold,trainingLabels_kfold);

prior0=sum(trainingLabels_kfold==0)/numel(trainingLabels_kfold);
prior1=sum(trainingLabels_kfold==1)/numel(trainingLabels_kfold);
prior2=sum(trainingLabels_kfold==2)/numel(trainingLabels_kfold);

u0=mean(trainingData_kfold(trainingLabels_kfold==0,:));
u1=mean(trainingData_kfold(trainingLabels_kfold==1,:));
u2=mean(trainingData_kfold(trainingLabels_kfold==2,:));


s0=std(trainingData_kfold(trainingLabels_kfold==0,:));
s1=std(trainingData_kfold(trainingLabels_kfold==1,:));
s2=std(trainingData_kfold(trainingLabels_kfold==2,:));

Y_kfold=[];
for i=1:size(testData_kfold,1)
    x=testData_kfold(i,:);
    for j=1:numel(x)
        G0(j)=myGaussianFCN(x(j),u0(j),s0(j));
        G1(j)=myGaussianFCN(x(j),u1(j),s1(j));
        G2(j)=myGaussianFCN(x(j),u1(j),s2(j));
    end
    P0=prior0*prod(G0)/(prod(G0)+prod(G1)+prod(G2));
    P1=prior1*prod(G1)/(prod(G0)+prod(G1)+prod(G2));
    P2=prior2*prod(G2)/(prod(G0)+prod(G1)+prod(G2));
    if double(P0>P1) & double(P0>P2 )
        Y_kfold(i,1)=0;
    elseif double(P1>P0) & double(P1>P2 )
        Y_kfold(i,1)=1;
    else
        Y_kfold(i,1)=2;
    end
end
%% Validation
Y_hat_kfold=predict(Model_kfold,testData_kfold);
acc_Y_hat_kfold(kfold_index,1)= sum(Y_hat_kfold==testLabels_kfold)/numel(testLabels_kfold)*100;
acc_Y_kfold(kfold_index,1)= sum(Y_kfold==testLabels_kfold)/numel(testLabels_kfold)*100;
end
%% display Accuracies o=in one plot for comparison
figure(1)
subplot(2,1,1)
hold on
kfold_index=1:KFold;
plot(kfold_index,acc_Y_hat_kfold,'o','MarkerSize',6)
ylabel('Acc-Y-hat-Average','fontsize',10)
xlabel('KFold','fontsize',10)
subplot(2,1,2)
hold on
plot(kfold_index,acc_Y_kfold,'r^','MarkerSize',6)
ylabel('Acc-Y-Average','fontsize',10)
xlabel('KFold','fontsize',10)
acc_Y_kfold_average=sum(acc_Y_kfold)/KFold
acc_Y_hat_kfold_average=sum(acc_Y_hat_kfold)/KFold


