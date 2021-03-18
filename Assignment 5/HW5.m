%% Prep
clear all; close all; clc
%% Monte Carlo
vidmp4 = VideoReader('monte_carlo_low.mp4');

video = read(vidmp4);
numFrames = get(vidmp4, 'NumFrames');
height = get(vidmp4, 'Height');
width = get(vidmp4, 'Width');
 

for j=1:numFrames
    vid_mat(:,j) = double(reshape(rgb2gray(video(:,:,:,j)), [], 1));
end

X = vid_mat(:,1:end-1);
Xshift = vid_mat(:,2:end);
r = 2;
t = linspace(0,numFrames, 2*numFrames);
dt = t(2) - t(1);

X_dmd = dmd(X,Xshift,r,dt);

 for j=1:numFrames - 1
 frame = reshape(X_dmd(:,j), height, width);
 frame = uint8(real(frame));
 imshow(frame); drawnow
 end
 
 foreground = X - X_dmd;
 for j=1:numFrames - 1
 frame = reshape(-foreground(:,j), height, width);
 frame = uint8(real(frame));
 imshow(frame); drawnow
 end
 
 %% Ski Drop
vidmp4 = VideoReader('ski_drop_low.mp4');

video = read(vidmp4);
numFrames = get(vidmp4, 'NumFrames');
height = get(vidmp4, 'Height');
width = get(vidmp4, 'Width');


for j=1:numFrames
    vid_mat(:,j) = double(reshape(rgb2gray(video(:,:,:,j)), [], 1));
end

X = vid_mat(:,1:end-1);
Xshift = vid_mat(:,2:end);
r = 2;
t = linspace(0,numFrames, 2*numFrames);
dt = t(2) - t(1);

X_dmd = dmd(X,Xshift,r,dt);

 for j=1:numFrames - 1
 frame = reshape(X_dmd(:,j), height, width);
 frame = uint8(real(frame));
 imshow(frame); drawnow
 end
 
 foreground = X - X_dmd;
 for j=1:numFrames - 1
 frame = reshape(-foreground(:,j), height, width);
 frame = uint8(real(frame));
 imshow(frame); drawnow
 end
 
%% function
function X_dmd = dmd(X,Xshift,r,dt)
[U, S, V] = svd(X, 'econ');
r = min(r, size(U,2));
U_r = U(:, 1:r);
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
A_tilde = U_r' * Xshift * V_r / S_r;
[W_r, D] = eig(A_tilde);
Phi = Xshift * V_r / S_r * W_r;
lambda = diag(D);
omega = log(lambda)/dt;
x1 =X(:, 1);
b = Phi\x1;
measurements_1 = size(X, 2);
time_dynamics = zeros(r, measurements_1);
t = (0:measurements_1 - 1).*dt;
for i = 1:measurements_1
    time_dynamics(:, i) = (b.*exp(omega*t(i)));
end
X_dmd = Phi * time_dynamics; % only background information
end