function [CORR, selection, imgs] = corrFunction(img, sel)

imgs = zeros(size(img));
imgs(img >= 127) = 255;
% selection = imgs(h(2):h(2)+h(4), h(1):h(1)+h(3));
selection = zeros(size(sel));
selection(sel >= 127) = 255;

FT = @(x) fftshift(fft2(x));
IFT = @(X) ifftshift(ifft2(X));

SI = size(imgs);
S = size(selection);

PAD = zeros(size(imgs));
PAD(1:S(1), 1:S(2)) = selection;
PAD = circshift(PAD, [floor(SI(1)/2)-floor(S(1)/2), floor(SI(2)/2)-floor(S(2)/2)]);

FT_IMG = FT(imgs);
FT_PAD = FT(PAD);

R = FT_IMG.*conj(FT_PAD);
CORR = abs(IFT(R));

end

    

    
    
    
fig, ax = plt.subplots(4, 4)
fig1, ax1 = plt.subplots(4, 4)
frame = 0

for i in range(4):
    for j in range(4):
        ax[i, j].imshow(data[frame, i*5+j])
        ax[i, j].axis('off')
        
        ax1[i, j].imshow(fit[frame, i*5+j])
        ax1[i, j].axis('off')
plt.show()