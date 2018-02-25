% Computer Project 2 Code

%Get data
trainObj = importdata("F:\\TAMU\\Spring 17\\ECEN 649 -PR\\Computer Project-2\train.txt",'\t');
train_x = trainObj.data;
train_y = train_x(:,72);
train_x = train_x(:,1:71);
trainHeader = trainObj.colheaders;

testObj = importdata("F:\\TAMU\\Spring 17\\ECEN 649 -PR\\Computer Project-2\\test.txt",'\t');
test_x = testObj.data;
test_y = test_x(:,72);
test_x = test_x(:,1:71);
testHeader = testObj.colheaders;

%Exhaustive search
fprintf("**************Exhaustive for size-1***********\n");
%size-1 feature set
%3NN
for i=2:1:size(train_x,2)
    new_train = [train_x(:,i)];
    mdl_knn = fitcknn(new_train,train_y,'NumNeighbors',3);
    new_train_label = mdl_knn.predict(new_train);
    
    if i>2
        if res_error > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_knn_1 = i;
        end       
    else
        res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
        best_knn_1 = 2;
    end   
end
best1 = char(trainHeader(best_knn_1));
fprintf('Best set of size-1 for 3NN is {%s} and the resubsitution error is %f\n',best1,res_error)

%DLDA
for i=2:1:size(train_x,2)
    new_train = [train_x(:,i)];
    mdl_dlda = fitcdiscr(new_train,train_y,'DiscrimType','diaglinear');
    new_train_label = predict(mdl_dlda, new_train);
    
    if i>2
        if res_error_dlda > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_dlda_1 = i;
        end
    else
        res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
        best_dlda_1 = 2;
    end
end

best1_dlda = char(trainHeader(best_dlda_1));
fprintf('Best set of size-1 for DLDA is {%s} and the resubsitution error is %f\n',best1_dlda,res_error_dlda)

%Test Set error calculation
best_train_knn = [train_x(:,best_knn_1)];
best_mdl_knn = fitcknn(best_train_knn, train_y,'NumNeighbors',3);
best_test_knn = [test_x(:,best_knn_1)];
test_label_knn = best_mdl_knn.predict(best_test_knn);
test_error_knn = sum(abs(test_label_knn - test_y))/size(test_x,1);
fprintf('Test set error for 3NN on best set {%s} of size-1 is %f\n',best1,test_error_knn);

best_train_dlda = [train_x(:,best_dlda_1)];
best_mdl_dlda = fitcdiscr(best_train_dlda, train_y, 'DiscrimType','diaglinear');
best_test_dlda = [test_x(:,best_dlda_1)];
test_label_dlda = predict(best_mdl_dlda, best_test_dlda);
test_error_dlda = sum(abs(test_label_dlda - test_y))/size(test_x,1);
fprintf('Test set error for DLDA on best set {%s} of size-1 is %f\n',best1_dlda,test_error_dlda);

fprintf("******************************************\n");

fprintf("**************Exhaustive for size-2***********\n");

%--------------------------------------------------------------------------%
%size-2 feature set
C = combnk(2:71,2);
%3NN
%error=zeros(1,size(C,1));
for i=1:1:size(C,1)
    new_train = [train_x(:,C(i,:))];
    mdl_knn = fitcknn(new_train,train_y,'NumNeighbors',3);
    new_train_label = mdl_knn.predict(new_train);    
    
    if i>1
        if res_error > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_knn_2 = i;
        end
        %error(1,i) = sum(abs(new_train_label - train_y))/size(train_x,1);
    else
        res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
        %error(1,i) = res_error;
        best_knn_2 = 1;
    end
end
best1 = char(trainHeader(C(best_knn_2,1)));
best2 = char(trainHeader(C(best_knn_2,2)));
fprintf('Best set of size-2 for 3NN is {%s, %s} and the resubstitution error is %f\n',best1,best2,res_error)

%DLDA

for i=1:1:size(C,1)
    new_train = [train_x(:,C(i,:))];   
    mdl_dlda = fitcdiscr(new_train,train_y,'DiscrimType','diagLinear');
    new_train_label = predict(mdl_dlda, new_train);    
    
    if i>1
        if res_error_dlda > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_dlda_2 = i;
        end
    else
        res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
        best_dlda_2 = 1;
    end   
end

best1_dlda = char(trainHeader(C(best_dlda_2,1)));
best2_dlda = char(trainHeader(C(best_dlda_2,2)));
fprintf('Best set of size-2 for DLDA is {%s, %s} and the resubsitution error is %f\n',best1_dlda,best2_dlda,res_error_dlda)

%Test Set error calculation
best_train_knn = [train_x(:,C(best_knn_2,:))];
best_mdl_knn = fitcknn(best_train_knn, train_y,'NumNeighbors',3);
best_test_knn = [test_x(:,C(best_knn_2,:))];
test_label_knn = best_mdl_knn.predict(best_test_knn);
test_error_knn = sum(abs(test_label_knn - test_y))/size(test_x,1);
fprintf('Test set error for 3NN on best set {%s, %s} of size-2 is %f\n',best1,best2,test_error_knn);

best_train_dlda = [train_x(:,C(best_dlda_2,:))];
best_mdl_dlda = fitcdiscr(best_train_dlda, train_y, 'DiscrimType','diaglinear');
best_test_dlda = [test_x(:,C(best_dlda_2,:))];
test_label_dlda = predict(best_mdl_dlda, best_test_dlda);
test_error_dlda = sum(abs(test_label_dlda - test_y))/size(test_x,1);
fprintf('Test set error for DLDA on best set {%s, %s} of size-2 is %f\n',best1_dlda,best2_dlda,test_error_dlda);

fprintf("******************************************\n");
    
fprintf("**************Exhaustive for size-3***********\n");
%--------------------------------------------------------------------------%
%size-3 feature set
C = combnk(2:71,3);

%3NN
for i=1:1:size(C,1)
    new_train = [train_x(:,C(i,:))];
    mdl_knn = fitcknn(new_train,train_y,'NumNeighbors',3);
    new_train_label = mdl_knn.predict(new_train);    
    
    if i>1
        if res_error > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_knn_3 = i;
        end
        %error(1,i) = sum(abs(new_train_label - train_y))/size(train_x,1);
    else
        res_error = sum(abs(new_train_label - train_y))/size(train_x,1);
        %error(1,i) = res_error;
        best_knn_3 = 1;
    end
end
best1 = char(trainHeader(C(best_knn_3,1)));
best2 = char(trainHeader(C(best_knn_3,2)));
best3 = char(trainHeader(C(best_knn_3,3)));
fprintf('Best set of size-3 for 3NN is {%s, %s, %s} and the resubstitution error is %f\n',best1,best2,best3,res_error)

%DLDA
for i=1:1:size(C,1)
    new_train = [train_x(:,C(i,:))];
    mdl_dlda = fitcdiscr(new_train,train_y,'DiscrimType','diaglinear');
    new_train_label = predict(mdl_dlda, new_train);
    
    if i>1
        if res_error_dlda > sum(abs(new_train_label - train_y))/size(train_x,1)
            res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
            best_dlda_3 = i;
        end
    else
        res_error_dlda = sum(abs(new_train_label - train_y))/size(train_x,1);
        best_dlda_3 = 1;
    end
end

best1_dlda = char(trainHeader(C(best_dlda_3,1)));
best2_dlda = char(trainHeader(C(best_dlda_3,2)));
best3_dlda = char(trainHeader(C(best_dlda_3,3)));
fprintf('Best set of size-3 for DLDA is {%s, %s, %s} and the resubsitution error is %f\n',best1_dlda,best2_dlda,best3_dlda,res_error_dlda)

%Test Set error calculation
best_train_knn = [train_x(:,C(best_knn_3,:))];
best_mdl_knn = fitcknn(best_train_knn, train_y,'NumNeighbors',3);
best_test_knn = [test_x(:,C(best_knn_3,:))];
test_label_knn = best_mdl_knn.predict(best_test_knn);
test_error_knn = sum(abs(test_label_knn - test_y))/size(test_x,1);
fprintf('Test set error for 3NN on best set {%s, %s, %s} of size-3 is %f\n',best1,best2,best3,test_error_knn);

best_train_dlda = [train_x(:,C(best_dlda_3,:))];
best_mdl_dlda = fitcdiscr(best_train_dlda, train_y, 'DiscrimType','diaglinear');
best_test_dlda = [test_x(:,C(best_dlda_3,:))];
test_label_dlda = predict(best_mdl_dlda, best_test_dlda);
test_error_dlda = sum(abs(test_label_dlda - test_y))/size(test_x,1);
fprintf('Test set error for DLDA on best set {%s, %s, %s} of size-3 is %f\n',best1_dlda,best2_dlda,best3_dlda,test_error_dlda);

fprintf("******************************************\n");

%--------------------------------------------------------------------------%
%Sequential forward Search
opts = statset('display','iter');
 
fprintf("**************Iteration for 3NN***********\n");
[fs,history] = sequentialfs(@my_crit_knn,train_x,train_y,'cv','resubstitution','nfeatures',8,'options',opts,'direction','forward');
fprintf("******************************************\n");
fprintf("**************Iteration for DLDA***********\n");

[fs1,history1] = sequentialfs(@my_crit_dlda,train_x,train_y,'cv','resubstitution','nfeatures',8,'options',opts,'direction','forward');
fprintf("******************************************\n");

%Test set estimate
%3NN
for i=1:1:size(history.In,1)
    new_train = [train_x(:,history.In(i,:)==1)];
    fprintf("{")
    features_selected = trainHeader(history.In(i,:)==1);
    for j=1:1:i-1
        fprintf("%s, ",char(features_selected(j)));
    end
    
    fprintf("%s} and the resub-error is %f\n",char(features_selected(i)),history.Crit(i));
    mdl_knn_fs  = fitcknn(new_train,train_y,'NumNeighbors',3);
    test_knn_fs = [test_x(:,history.In(i,:)==1)];
    test_error_knn_fs = sum(abs(mdl_knn_fs.predict(test_knn_fs) - test_y))/size(test_x,1);
    fprintf("Test error of KNN with set size-%d is %f\n\n",i,test_error_knn_fs);
end

fprintf("******************************************\n");

%Test Set estimate
%DLDA
for i=1:1:size(history1.In,1)
    new_train = [train_x(:,history1.In(i,:)==1)];
    fprintf("{")
    features_selected_dlda = trainHeader(history1.In(i,:)==1);
    for j=1:1:i-1
        fprintf("%s, ",char(features_selected_dlda(j)));
    end
    fprintf("%s} and the resub-error is %f\n",char(features_selected_dlda(i)),history1.Crit(i));
    mdl_dlda_fs  = fitcdiscr(new_train,train_y,'DiscrimType','diaglinear');
    test_dlda_fs = [test_x(:,history1.In(i,:)==1)];
    test_error_dlda_fs = sum(abs(predict(mdl_dlda_fs,test_dlda_fs) - test_y))/size(test_x,1);
    fprintf("Test error of DLDA with set size-%d is %f\n\n",i,test_error_dlda_fs);
end

function val_knn = my_crit_knn(xT,yT,xt,yt)
    mdl_knn = fitcknn(xT,yT,'NumNeighbors',3);    
    val_knn = sum(abs(predict(mdl_knn,xt) - yt)) ;   
end

function val_dlda = my_crit_dlda(xT,yT,xt,yt)
    mdl_dlda = fitcdiscr(xT,yT,'DiscrimType','diaglinear');    
    val_dlda = sum(abs(predict(mdl_dlda,xt) - yt)) ;   
end






