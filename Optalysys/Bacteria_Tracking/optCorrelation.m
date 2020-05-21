clear
clc
set(0, 'DefaultFigureColormap',feval('gray'));
img = imread('Input_Bacteria.png');

f = figure(1);
f. WindowState = 'maximized';
imagesc(img);
axis square; 
colormap gray;
h = floor(getrect);
close all

%% Convert to binary
imgs = zeros(size(img));
imgs(img >= 127) = 255;
selection = imgs(h(2):h(2)+h(4), h(1):h(1)+h(3));

%% Convert to uint8
in = uint8(imgs);
filt = uint8(selection);
%% Compute correlation in optical computer
corr = BacteriaScan(in, filt);

%%
M = max(corr(:));
[id, jd] = find(flipdim(corr, 1) == M);

%%
figure(2)
mesh(flipdim(corr,1))
hold on
M = max(corr(:));
scatter3(jd, id, double(M)*ones(size(id)), 100, 'r')