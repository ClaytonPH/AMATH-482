%% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata 5
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
%% Averaging
ave = zeros(n,n,n);

for j=1:49 % Averages all 49 instances of Fourier Transformation
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    utn = fftn(Un);
    ave = ave + utn;
end
ave = abs(fftshift(ave))/49;
[M, ind] = max(abs(ave),[],'all','linear');
close all, isosurface(Kx,Ky,Kz,abs(ave)/M,0.7)
axis([-10 10 -10 10 -10 10]); grid on, drawnow
[x0, y0, z0] = ind2sub([64 64 64], ind);
% Finds center frequency
center_Kx = Kx(x0,y0,z0);
center_Ky = Ky(x0,y0,z0);
center_Kz = Kz(x0,y0,z0);
%% Filtering
tau = 1;
% Gaussian Filter
filter = fftshift(exp(tau*(-(Kx-center_Kx).^2 - (Ky-center_Ky).^2)-(Kz-center_Kz).^2)); 
pxyz = zeros(3,49);
% Applies the Gaussian filter and finds the coordinates of the sub for each time
for j=1:49 
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    utn = ifftn(fftn(Un).*filter);
    [M, ind] = max(abs(utn),[],'all', 'linear');
    [xc, yc, zc] = ind2sub([64 64 64], ind);
    pxyz(1,j) = Kx(xc,yc,zc);
    pxyz(2,j) = Ky(xc,yc,zc);
    pxyz(3,j) = Kz(xc,yc,zc);
end 
plot3(pxyz(1,:),pxyz(2,:),pxyz(3,:))
grid on