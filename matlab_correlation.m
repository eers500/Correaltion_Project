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

SI = size(IM);
S = size(IFILT);
DSY = SI(1) - S(1);
DSX = SI(2) - S(2);

if mod(DSY,2) == 0 && mod(DSX,2) == 0
    NY = DSY / 2;
    NX = DSX / 2;
    IPAD = padarray(IFILT, [NY,NX]);
%     IPAD = np.pad(IFILT, ((NY, NY), (NX, NX)), 'constant', constant_values=0);
elseif mod(DSY,2) == 1 && mod(DSX,2) == 1
    NY = int(floor(DSY / 2));
    NX = int(floor(DSX / 2));
%     IPAD = np.pad(IFILT, ((NY, NY + 1), (NX, NX + 1)), 'constant', constant_values=0);
elseif mod(DSY,2) == 0 && mod(DSX,2) == 1
    NY = DSY / 2;
    NX = floor(DSX / 2);
%     IPAD = np.pad(IFILT, ((NY, NY), (NX, NX + 1)), 'constant', constant_values=0);
elseif mod(DSY,2) == 1 && mod(DSX,2) == 0
    NY = floor(DSY / 2);
    NX = DSX / 2;
    IPAD = padarray(IFILT,[NY+1,NX]);
    IPAD(end,:) = [];
end

I_FT = FT(IM);
IFILT_FT = IFT(IPAD);

R = I_FT*conj(IFILT_FT);
r = real(IFT(R));
%%
subplot(2,2,1); imagesc(IM); colormap gray; axis square
subplot(2,2,2); imagesc(IPAD); colormap gray; axis square
subplot(2,2,3); imagesc(CORR); colormap gray; axis square
subplot(2,2,4); imagesc(r); colormap gray; axis square


