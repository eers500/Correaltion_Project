clear all
clc

img = imread('Input_Bacteria.png');

f = figure(1);
f. WindowState = 'maximized';
imagesc(img);
axis square; 
colormap gray;
h = floor(getrect);
close all
selection = img(h(2):h(2)+h(4), h(1):h(1)+h(3));

%% Test correaltion
% clear all
% clc
% img = imread('8.png');
% selection = imread('Filter.png');

%% Run correlation in CPU
tic
[CORR, SEL,IM_BIN] = corrFunction(img, selection);
toc
M = max(CORR(:));
[id, jd] = find(CORR == M);

%% Run correlation of Optical Computer
tic
[CORRO, SELO, IMO_BIN] = optCorrFunction(img, selection);
toc
MO = max(CORRO(:));
[ido, jdo] = find(CORRO == MO);

%% Plot CPU correlation results
figure(1)
subplot(1, 3, 1)
sgtitle('Correlation')
imagesc(img); hold on
scatter(jd, id, 100, 'r');
axis square

subplot(1,3,2)
imagesc(SEL); 
axis square

subplot(1,3,3)
imagesc(CORR); hold on;
scatter(jd, id, 100, 'r');
axis square


%% Plot Optical Computer correlation results
figure(2)
subplot(1, 3, 1)
sgtitle('Optical Correlation')
imagesc(img); hold on
scatter(jd, id, 100, 'r');
axis square

subplot(1,3,2)
imagesc(SELO); 
axis square

subplot(1,3,3)
imagesc(CORRO); hold on;
scatter(jdo, ido, 100, 'r');
axis square
