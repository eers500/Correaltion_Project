function [filt] = createFilter(img, shape)


% Convert to binary
%selection = zeros(size(sel));
%selection(sel >= 127) = 255;

% PAD
SI = shape;
S = size(img);
PAD = zeros(SI);
PAD(1:S(1), 1:S(2)) = img;
PAD = circshift(PAD, [floor(SI(1)/2)-floor(S(1)/2), floor(SI(2)/2)-floor(S(2)/2)]);

% Phase
FT = @(x) ifftshift(fft2(fftshift(x)));
phase_sel = exp(1i * pi * PAD / 255);
filter = FT(phase_sel);
f = -255.*angle(filter);

ff = zeros(size(f));
ff(f >= 0) = 255;

% Convert to uint8
filt = uint8(ff);

end