%% Prep
clear all; close all; clc
load cam1_4.mat;
load cam2_4.mat;
load cam3_4.mat;
load cam1_3.mat;
load cam2_3.mat;
load cam3_3.mat;
load cam1_2.mat;
load cam2_2.mat;
load cam3_2.mat;
load cam1_1.mat;
load cam2_1.mat;
load cam3_1.mat;
%% Test 1 Setup
vidCrop1_1 = vidFrames1_1(220:390,315:385,:,:);
vidCrop2_1 = vidFrames2_1(100:320,260:340,:,12:237);
vidCrop3_1 = vidFrames3_1(245:310,275:440,:,1:226);
vid1_1x = zeros(1,226);
vid1_1y = zeros(1,226);
vid2_1x = zeros(1,226);
vid2_1y = zeros(1,226);
vid3_1x = zeros(1,226);
vid3_1y = zeros(1,226);
for j = 1:226
    bwFrame = rgb2gray(vidCrop1_1(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid1_1x(j), vid1_1y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop2_1(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid2_1x(j), vid2_1y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop3_1(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid3_1x(j), vid3_1y(j)] = ind2sub(size(bwFrame), ind);
end
%% Test 1 SVD
ses1 = [vid1_1x; vid1_1y; vid2_1x; vid2_1y; vid3_1x; vid3_1y];
[m,n] = size(ses1);
avg = mean(ses1,2);
ses1 = ses1-repmat(avg,1,n);
[U,S,V] = svd(ses1'/sqrt(n-1),'econ');
sig = diag(S);

figure(1)
subplot(1,2,1)
plot(100*sig/sum(sig),'O')
title("Test 1: Energy of Each Singular Value")
xlabel("Singular Value")
ylabel("Percent of Total Energy")
subplot(3,2,2)
plot(U(:,1))
title("Test 1: Position of Can - First Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,4)
plot(U(:,2))
title("Test 1: Position of Can - Second Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,6)
plot(U(:,3))
title("Test 1: Position of Can - Third Mode")
xlabel("Frame")
ylabel("Position")
%% Test 2 Setup
vidCrop1_2 = vidFrames1_2(225:400,300:400,:,:);
vidCrop2_2 = vidFrames2_2(65:350,200:375,:,26:341);
vidCrop3_2 = vidFrames3_2(200:325,275:450,:,6:319);
vid1_2x = zeros(1,314);
vid1_2y = zeros(1,314);
vid2_2x = zeros(1,314);
vid2_2y = zeros(1,314);
vid3_2x = zeros(1,314);
vid3_2y = zeros(1,314);
for j = 1:314
    bwFrame = rgb2gray(vidCrop1_2(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid1_2x(j), vid1_2y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop2_2(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid2_2x(j), vid2_2y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop3_2(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid3_2x(j), vid3_2y(j)] = ind2sub(size(bwFrame), ind);
end
%% Test 2 SVD
ses2 = [vid1_2x; vid1_2y; vid2_2x; vid2_2y; vid3_2x; vid3_2y];
[m,n] = size(ses2);
avg = mean(ses2,2);
ses2 = ses2-repmat(avg,1,n);
[U,S,V] = svd(ses2'/sqrt(n-1),'econ');
sig = diag(S);

figure(2)
subplot(1,2,1)
plot(100*sig/sum(sig),'O')
title("Test 2: Energy of Each Singular Value")
xlabel("Singular Value")
ylabel("Percent of Total Energy")
subplot(3,2,2)
plot(U(:,1))
title("Test 2: Position of Can - First Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,4)
plot(U(:,2))
title("Test 2: Position of Can - Second Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,6)
plot(U(:,3))
title("Test 2: Position of Can - Third Mode")
xlabel("Frame")
ylabel("Position")
%% Test 3 Setup
vidCrop1_3 = vidFrames1_3(225:380,280:380,:,7:239);
vidCrop2_3 = vidFrames2_3(135:350,210:390,:,32:264);
vidCrop3_3 = vidFrames3_3(200:325,275:400,:,1:233);
vid1_3x = zeros(1,233);
vid1_3y = zeros(1,233);
vid2_3x = zeros(1,233);
vid2_3y = zeros(1,233);
vid3_3x = zeros(1,233);
vid3_3y = zeros(1,233);
for j = 1:233
    bwFrame = rgb2gray(vidCrop1_3(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid1_3x(j), vid1_3y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop2_3(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid2_3x(j), vid2_3y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop3_3(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid3_3x(j), vid3_3y(j)] = ind2sub(size(bwFrame), ind);
end
%% Test 3 SVD
ses3 = [vid1_3x; vid1_3y; vid2_3x; vid2_3y; vid3_3x; vid3_3y];
[m,n] = size(ses3);
avg = mean(ses3,2);
ses3 = ses3-repmat(avg,1,n);
[U,S,V] = svd(ses3'/sqrt(n-1),'econ');
sig = diag(S);

figure(3)
subplot(1,2,1)
plot(100*sig/sum(sig),'O')
title("Test 3: Energy of Each Singular Value")
xlabel("Singular Value")
ylabel("Percent of Total Energy")
subplot(3,2,2)
plot(U(:,1))
title("Test 3: Position of Can - First Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,4)
plot(U(:,2))
title("Test 3: Position of Can - Second Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,6)
plot(U(:,3))
title("Test 3: Position of Can - Third Mode")
xlabel("Frame")
ylabel("Position")
%% Test 4 Setup
vidCrop1_4 = vidFrames1_4(225:360,320:450,:,:);
vidCrop2_4 = vidFrames2_4(95:310,220:410,:,6:397);
vidCrop3_4 = vidFrames3_4(175:275,300:460,:,1:392);
vid1_4x = zeros(1,392);
vid1_4y = zeros(1,392);
vid2_4x = zeros(1,392);
vid2_4y = zeros(1,392);
vid3_4x = zeros(1,392);
vid3_4y = zeros(1,392);
for j = 1:392
    bwFrame = rgb2gray(vidCrop1_4(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid1_4x(j), vid1_4y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop2_4(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid2_4x(j), vid2_4y(j)] = ind2sub(size(bwFrame), ind);
    
    bwFrame = rgb2gray(vidCrop3_4(:,:,:,j));
    [M, ind] = max(bwFrame(:));
    [vid3_4x(j), vid3_4y(j)] = ind2sub(size(bwFrame), ind);
end
%% Test 4 SVD
ses4 = [vid1_4x; vid1_4y; vid2_4x; vid2_4y; vid3_4x; vid3_4y];
[m,n] = size(ses4);
avg = mean(ses4,2);
ses4 = ses4-repmat(avg,1,n);
[U,S,V] = svd(ses4'/sqrt(n-1),'econ');
sig = diag(S);

figure(4)
subplot(1,2,1)
plot(100*sig/sum(sig),'O')
title("Test 4: Energy of Each Singular Value")
xlabel("Singular Value")
ylabel("Percent of Total Energy")
subplot(3,2,2)
plot(U(:,1))
title("Test 4: Position of Can - First Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,4)
plot(U(:,2))
title("Test 4: Position of Can - Second Mode")
xlabel("Frame")
ylabel("Position")
subplot(3,2,6)
plot(U(:,3))
title("Test 4: Position of Can - Third Mode")
xlabel("Frame")
ylabel("Position")