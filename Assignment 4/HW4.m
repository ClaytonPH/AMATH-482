%% Prep
clear all; close all; clc
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

%% Reshape 
X = zeros(784,60000);
for j = 1:60000
    img = transpose(images(:,:,j));
    X(:,j) = img(:); 
end

ind0 = find(labels == 0);
ind1 = find(labels == 1);
ind2 = find(labels == 2);
ind3 = find(labels == 3);
ind4 = find(labels == 4);
ind5 = find(labels == 5);
ind6 = find(labels == 6);
ind7 = find(labels == 7);
ind8 = find(labels == 8);
ind9 = find(labels == 9);

[label, idx] = sort(labels);
trainImg = X(:,idx);
%% SV Spectrum
[m,n] = size(trainImg);
avg = mean(trainImg,2);
trainImg = trainImg - repmat(avg,1,n);
[U,S,V] = svd(trainImg,'econ');
sig = diag(S);
figure(1)
plot(100*sig/sum(sig),'O')
sum(sig(1:70))/sum(sig)
title("Singular Value Spectrum")
xlabel("Singular Value")
ylabel("Percent Of Total Information")
%% 4
figure(2)
view(3)
hold on
plot3(U(:,2)'*trainImg(:, 1:5923),U(:,3)'*trainImg(:,1:5923),U(:,5)'*trainImg(:,1:5923), 'blackO')
plot3(U(:,2)'*trainImg(:, 5924:12665),U(:,3)'*trainImg(:,5924:12665),U(:,5)'*trainImg(:,5924:12665), 'O')
plot3(U(:,2)'*trainImg(:, 12666:18623),U(:,3)'*trainImg(:,12666:18623),U(:,5)'*trainImg(:,12666:18623), 'O')
plot3(U(:,2)'*trainImg(:, 18624:24753),U(:,3)'*trainImg(:,18624:24753),U(:,5)'*trainImg(:,18624:24753), 'O')
plot3(U(:,2)'*trainImg(:, 24754:30596),U(:,3)'*trainImg(:,24754:30596),U(:,5)'*trainImg(:,24754:30596), 'O')
plot3(U(:,2)'*trainImg(:, 30597:36017),U(:,3)'*trainImg(:,30597:36017),U(:,5)'*trainImg(:,30597:36017), 'd')
plot3(U(:,2)'*trainImg(:, 36018:41935),U(:,3)'*trainImg(:,36018:41935),U(:,5)'*trainImg(:,36018:41935), 'd')
plot3(U(:,2)'*trainImg(:, 41936:48200),U(:,3)'*trainImg(:,41936:48200),U(:,5)'*trainImg(:,41936:48200), 'd')
plot3(U(:,2)'*trainImg(:, 48201:54051),U(:,3)'*trainImg(:,48201:54051),U(:,5)'*trainImg(:,48201:54051), 'd')
plot3(U(:,2)'*trainImg(:, 54052:60000),U(:,3)'*trainImg(:,54052:60000),U(:,5)'*trainImg(:,54052:60000), 'd')
title("Columns Of V Colored By Digit")
xlabel("X Axis")
ylabel("Y Axis")
zlabel("Z Axis")
legend('0','1','2','3','4','5','6','7','8','9')

%% Testing Prep
Y = zeros(784,10000);
for j = 1:10000
    img = transpose(testImages(:,:,j));
    Y(:,j) = img(:); 
end

indTest0 = find(testLabels == 0);
indTest1 = find(testLabels == 1);
indTest2 = find(testLabels == 2);
indTest3 = find(testLabels == 3);
indTest4 = find(testLabels == 4);
indTest5 = find(testLabels == 5);
indTest6 = find(testLabels == 6);
indTest7 = find(testLabels == 7);
indTest8 = find(testLabels == 8);
indTest9 = find(testLabels == 9);

[label2, idy] = sort(testLabels);
testImg = Y(:,idy);
testDigits = U(:,1:70)'*testImg;
%% LDA 0 and 1
class = classify(testDigits(:,1:(size(indTest0)+size(indTest1)))', V(1:12665,1:70), label(1:12665)');
plot(class, 'o')
title("LDA For Two Digits (0 and 1)")
xlabel("Projection")
ylabel("Assigned Digit (0 or 1)")
count = 0;
for j = 1:2115
    if class(j) == label2(j)
        count = count + 1;
    end
end
percent = count/2115
%% LDA 0, 1, and 2
testDigits = U(:,1:70)'*testImg;
class = classify(testDigits(:,1:(size(indTest0)+size(indTest1)+size(indTest2)))', V(1:18623,1:70), label(1:18623)');
plot(class, 'o')
title("LDA For Three Digits (0, 1, and 2)")
xlabel("Projection")
ylabel("Assigned Digit (0,1, or 2)")
count = 0;
for j = 1:3147
    if class(j) == label2(j)
        count = count + 1;
    end
end
percent = count/3147
%% LDA 1 and 2
testDigits = U(:,1:70)'*testImg;
class = classify(testDigits(:,981:3147)', V(5924:18623,1:70), label(5924:18623)');
plot(class, 'o')
title("LDA For Three Digits (0, 1, and 2)")
xlabel("Projection")
ylabel("Assigned Digit (0,1, or 2)")
count = 0;
newlabel = label2(981:3147);
for j = 1:2167
    if class(j) == newlabel(j)
        count = count + 1;
    end
end
percent = count/2167
%% Tree for all digits
tree = fitctree(V(:,1:70),label,'MaxNumSplits',10);
testTree = predict(tree, testDigits');
count = 0;
for j = 1:10000
    if testTree(j) == label2(j)
        count = count + 1;
    end
end
percent = count/10000
%% Tree for 0 and 1
tree=fitctree(V(1:12665,1:70),label(1:12665),'MaxNumSplits',2);
testTree = predict(tree, testDigits');
count = 0;
for j = 1:2115
    if testTree(j) == label2(j)
        count = count + 1;
    end
end
percent = count/2115
%% Tree for 1 and 2
tree=fitctree(V(5924:18623,1:70),label(5924:18623),'MaxNumSplits',2);
testTree = predict(tree, testDigits(:,981:3147)');
count = 0;
newlabel = label2(981:3147);
for j = 1:2167
    if testTree(j) == newlabel(j)
        count = count + 1;
    end
end
percent = count/2167

%% SVM for 0 and 1
Mdl = fitcsvm(V(1:12665,1:70),label(1:12665));
test_labels = predict(Mdl,testDigits');
count = 0;
for j = 1:2115
    if test_labels(j) == label2(j)
        count = count + 1;
    end
end
percent = count/2115

%% SVM for 1 and 2
Mdl = fitcsvm(V(5924:18623,1:70),label(5924:18623));
test_labels = predict(Mdl,testDigits(:,981:3147)');
count = 0;
newlabel = label2(981:3147);
for j = 1:2167
    if test_labels(j) == newlabel(j)
        count = count + 1;
    end
end
percent = count/2167

%% SVM for all digits
Mdl = fitcecoc(V(:,1:70),label);
test_labels = predict(Mdl,testDigits');
count = 0;
for j = 1:10000
    if test_labels(j) == label2(j)
        count = count + 1;
    end
end
percent = count/10000