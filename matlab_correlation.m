clear
IM = imread('NORM_MF1_30Hz_200us_awaysection.png');
IFILT = double(imread('MF1_1.png'));
IFILTPAD =uint8(imread('MF1_PAD.png'));
IFILTPAD = IFILTPAD(:, :, 1);
IFILTPAD(IFILTPAD == 68) = 0;
%%
clear
IM = imread('R.png');
IFILT = double(imread('SampleRing.png'));
% IFILTPAD = double(imread('MF1_PAD.png'));
% IFILTPAD = IFILTPAD(:, :, 1);

%%
% CORR = imfilter(IM, IFILT, 'symmetric', 'corr');
CORR = xcorr2(IM, IFILTPAD);
%%
FT = @(x)  ifftshift(fft2(fftshift( x )));
IFT = @(X) ifftshift(ifft2(fftshift( X )));

I_FT = FT(IM);
IFILT_FT = IFT(IFILTPAD);

R = I_FT*conj(IFILT_FT);
r = real(IFT(R));
%%
subplot(2,2,1); imagesc(IM); colormap gray; axis square
subplot(2,2,2); imagesc(IFILT); colormap gray; axis square
subplot(2,2,3); imagesc(IFILTPAD); colormap gray; axis square
subplot(2,2,4); imagesc(r); colormap gray; axis square


