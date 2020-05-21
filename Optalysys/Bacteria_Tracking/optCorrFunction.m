function [corr, filt, imgs] = optCorrFunction(img, sel)

% Convert to binary
imgs = zeros(size(img));
imgs(img >= 127) = 255;
selection = zeros(size(sel));
selection(sel >= 127) = 255;

% PAD
SI = size(imgs);
S = size(selection);
PAD = zeros(size(imgs));
PAD(1:S(1), 1:S(2)) = selection;
PAD = circshift(PAD, [floor(SI(1)/2)-floor(S(1)/2), floor(SI(2)/2)-floor(S(2)/2)]);

% Phase
FT = @(x) ifftshift(fft2(fftshift(x)));
phase_sel = exp(1i * pi * PAD / 255);
filter = FT(phase_sel);
f = -255.*angle(filter);

ff = zeros(size(f));
ff(f >= 0) = 255;
% Convert to uint8
in = uint8(imgs);
filt = uint8(ff);

% Compute correlation in optical computer
corr = BacteriaScan(in, filt);

end