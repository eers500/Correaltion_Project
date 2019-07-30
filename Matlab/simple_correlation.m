%% A first version of a 4f correlator simulator
function[] = simple_correlation(input)
close all

if nargin<1
    disp(['Input png image required.']);
    clear
    return
end
input = uint8(imread(input));
input = input(1:1000, 1:1000); % Make the input the same size as the filter (1000x1000) for simulation.
                               % Not necessary for optics which can be up to 1080x1920 
binary_input = zeros(size(input));                               
binary_input(input>mean(input)) = 255; % Make input binary (black and white, not greyscale) to improve contrast
        
% Select target bacteria from image
figure('Position', [100,100,700,700]);
colormap gray
imagesc(binary_input);
[X,Y] = ginput(1);
windowHalfSize = 30;
ref = binary_input(Y-windowHalfSize:Y+windowHalfSize-1, X-windowHalfSize:X+windowHalfSize-1);
figure(2);
imagesc(ref)
%ref   = uint8(imread(target));   % Or read it in from a png
ref = ref(:,:,1);                 % Flatten

% Check size is even (prevent non-integers when padding.
if rem(size(ref,1),2)
   ref(end,:) = [];
end
if rem(size(ref,2),2)
   ref(:,end) = [];
end
reference = zeros(size(binary_input)); % Make a padded (1000x1000) reference image 
center = floor(size(reference)/2);
halfTargetSize = floor(size(ref)/2);
% Insert the target image in to the center of the filter
reference(center(1)-halfTargetSize(1):center(1)+halfTargetSize(1)-1, center(2)-halfTargetSize(1):center(2)+halfTargetSize(1)-1) = ref;
    
figure(1);
subplot(2,1,1);
imshow(binary_input);
daspect([1,1,1]);
subplot(2,1,2);
imshow(reference);
%input = input(1:length(reference), 1:length(reference));
binary_input = double(binary_input(:, :, 1));
binary_input = fliplr(binary_input);


reference = double(reference(:, :, 1));
reference = fliplr(reference);
%% Pad the input and reference with black border to make 2x image.
%% The purpose of doing this is to have finer grained Fourier coefficients for a more accurate FFT and convolution calculation
%% input = pad2x(input)
%% reference = pad2x(reference)

%% Define forwards and inverse Fourier transforms 
FT = @(x)  ifftshift(fft2(fftshift( x )));
IFT = @(X) ifftshift(ifft2(fftshift( X )));

phase_reference = exp(1i * pi * reference / 255);
%phase_reference = 1 + imag(phase_reference);
phase_input     = exp(1i * pi * binary_input / 255);
%phase_input = 1 + imag(phase_input);

figure(2);
subplot(2,1,1);
imagesc(imag(phase_input));     daspect([1,1,1]);
title('Phase Input');
subplot(2,1,2);
imagesc(imag(phase_reference)); daspect([1,1,1]);
title('Phase Reference');



%% Convert the reference to a filter
%Phase_Reference = phase_reference;     % If the filter is already FFTd. 
Phase_Reference = FT(phase_reference);  
Phase_Input = FT(phase_input);

%% <Optional> Ouput reference filter.
%% At this point output an image of the reference filter, either to send to the SLM or to save as an image for later.
%% Output_Phase_Reference = abs(angle(Phase_Reference))  -- map angle to [0, pi]
%% Output_Phase_Reference = fftshift(Output_Phase_Reference) -- assume humans want centered zero order, what about for output to SLM?
%% save_image(Output_Phase_Reference*255/pi)

%% Emulate the correlation process
%% We use abs here because our filter SLM is a phase SLM that can represent [0, pi]
%output = IFT( Phase_Input .* exp(-1i * abs(angle(Phase_Reference))));

% Usual Method of multiplication
% Calculate the phase retardation of the filter SLM. 
Phase_Filter = exp(-1i * abs(angle(Phase_Reference)));
% Multiplication
outputStd = IFT(Phase_Input .* Phase_Filter);
output_std_intensity = abs(outputStd).^2;

% Log Addition Method
% LogInput  = log(Phase_Input);
% LogFilter = log(Phase_Filter);
% outputLog = IFT(exp(LogInput + LogFilter));
% output_log_intensity = abs(outputLog).^2;








%% Need to fftshift to match locations on the original image.
%% I'm not actually sure why this is needed?
%output = fftshift(output);

%% Undo any padding that was applied. That is, strip off black border.
%% output =unpad2x(output)


%% Display the output
%% figure; imagesc(output_intensity)
figure(3);
%subplot(1,2,1);
mesh(output_std_intensity);  
title('Standard Product Method');

Filter = -255.*angle(Phase_Reference);


% Save images.
% Save binary input, target filter image
% and binary input image showing the chosen bacteria.

% Save filter image with user defined filename
fname = inputdlg('Save filter as...')
fullName = strcat(fname{1}, '[',num2str(round(X)),',',num2str(round(Y)),'].png');
imwrite(Filter, fullName);

% Save binary input image for optics
imwrite(binary_input, 'Binary_Input.png');
disp(['Saved Images as : Binary_Input.png and ', fullName]);


% Save annotated input image
h = figure(3);
annotatedInput = input;
colormap gray
imagesc(annotatedInput)
hold on
scatter(X,Y, 200, [1,0,0]);
saveas(h, ['Input_Image_[', num2str(round(X)),',',num2str(round(Y)),'].png'], 'png');
disp('Done')

%% Save the output to an 8-bit image file
%% What is the best way to do this??
%% For now:
%% scaled_output_intensity = output_intensity .* 255/(max(output_intensity))
%% save_image(scaled_output_intensity)