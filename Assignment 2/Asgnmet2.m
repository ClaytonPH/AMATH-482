%% Loading Songs
clear all; close all; clc

[GNR, GNRfs] = audioread('GNR.m4a');
tr_gnr = length(GNR)/GNRfs; % record time in seconds

[PF, PFfs] = audioread('Floyd.m4a');
PF = PF(1:end-1); % odd length
tr_PF = length(PF)/PFfs; % record time in seconds
tPF = PF(1:length(PF)/4);
ttr_PF = length(tPF)/PFfs;


%% Guns and Roses

a = 100;
L = tr_gnr; n = length(GNR);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1  -n/2:-1];
ks = fftshift(k);
tslide = 0:0.1:tr_gnr;
spec = zeros(length(tslide),length(GNR));
maxFreq = zeros(length(tslide),1);
for j = 1:length(tslide)
    gaus = exp(-a*(t-tslide(j)).^2);  
    gab = gaus.*transpose(GNR);
    gabt = fft(gab);
    spec(j,:) = abs(fftshift(gabt));
    AbsSgt = fftshift(abs(gabt));
    [maxv, idx] = max(AbsSgt);
    maxFreq(j) = ks(idx);
end

figure(1)
pcolor(tslide, ks, log((abs(spec.')+1))), shading interp, colormap hot
set(gca, 'Ylim', [0 1000])
title('Sweet Child O Mine', 'Fontsize', 12)
xlabel('Time (t)', 'Fontsize', 12)
ylabel('Frequency (Hz)', 'Fontsize', 12)
notes = [276 553 415 369 310 702 742];

%% Pink Floyd

a = 100;
L = ttr_PF; 
n = length(tPF);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1  -n/2:-1];
ks = fftshift(k);
tslide = 0:0.1:ttr_PF;
spec = zeros(length(tslide),n);
maxFreq = zeros(length(tslide),1);
for j = 1:length(tslide)
    gaus = exp(-a*(t-tslide(j)).^2);  %Gabor
    gab=gaus.*transpose(tPF);
    gabt = fft(gab);
    spec(j,:) = abs(fftshift(gabt));
    AbsSgt = fftshift(abs(gabt));
    [M, ind] = max(AbsSgt);
    maxFreq(j) = ks(ind);
end
figure(2)
pcolor(tslide, ks, log((abs(spec.')+1))), shading interp, colormap hot
set(gca, 'Ylim', [0 1000])
title('Comfortably Numb', 'Fontsize', 12)
xlabel('Time (t)', 'Fontsize', 12)
ylabel('Frequency (Hz)', 'Fontsize', 12)
notes = [124 110 98 90 83];


%% PF Bass Filtered

tau = 1;
spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    gaus = exp(-a*(t-tslide(j)).^2);  %Gabor
    gab=gaus.*transpose(tPF);
    gabt = fft(gab).*fftshift(exp(-tau*(ks-abs(maxFreq(j))).^2));
    spec(j,:) = abs(fftshift(gabt));
end

figure(3)
pcolor(tslide, ks, log((abs(spec.')+1))), shading interp, colormap hot
set(gca, 'Ylim', [50 250])
title('Filtered Comfortably Numb Bass', 'Fontsize', 12)
xlabel('Time (t)', 'Fontsize', 12)
ylabel('Frequency (Hz)', 'Fontsize', 12)

%% PF Guitar

L = tr_PF;
n = length(PF);
k = (1/L)*[0:n/2-1  -n/2:-1];
ks = fftshift(k);
FPF = fftshift(fft(PF));
for j = 1:length(ks)
    if abs(ks(j)) < 150
        FPF(j) = 0;
    end
end
nPF = ifft(fftshift(FPF));
%% PF Bass Removed

tPF = nPF(1:length(nPF)/4);
ttr_PF = length(tPF)/PFfs;
a = 100;
L = ttr_PF; n=length(tPF);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1  -n/2:-1];
ks = fftshift(k);
tslide = 0:0.1:ttr_PF;
spec = zeros(length(tslide),n);
maxFreq = zeros(length(tslide),1);
for j = 1:length(tslide)
    gaus = exp(-a*(t-tslide(j)).^2);  %Gabor
    gab=gaus.*transpose(tPF);
    gabt = fft(gab);
    spec(j,:) = abs(fftshift(gabt));
    AbsSgt = fftshift(abs(gabt));
    [M, ind] = max(AbsSgt);
    maxFreq(j) = ks(ind);
end
figure(4)
pcolor(tslide, ks, log((abs(spec.')+1))), shading interp, colormap hot
set(gca, 'Ylim', [0 1000])
title('Comfortably Numb Bass Removed', 'Fontsize', 12)
xlabel('Time (t)', 'Fontsize', 12)
ylabel('Frequency (Hz)', 'Fontsize', 12)

%% PF Guitar Filtered

tau = 1;
spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    gaus = exp(-a*(t-tslide(j)).^2);  %Gabor
    gab=gaus.*transpose(tPF);
    gabt = fft(gab).*fftshift(exp(-tau*(ks-abs(maxFreq(j))).^2));
    spec(j,:) = abs(fftshift(gabt));
end

figure(5)
pcolor(tslide, ks, log((abs(spec.')+1))), shading interp, colormap hot
set(gca, 'Ylim', [150 1000])
title('Filtered Comfortably Guitar Filtered', 'Fontsize', 12)
xlabel('Time (t)', 'Fontsize', 12)
ylabel('Frequency (Hz)', 'Fontsize', 12)