function[] =  batchDemo()

close all
clear all
pause(1)

plotStuff = true;
addpath('./common');

FT = @(x) ifftshift(fft2(fftshift(x)));

 %% Connect to the hardware
if ~exist('camera','var') || ~exist('slm','var') || exist('json_file','var')
	[camera,slm,json_file] = setupOpticalCorrelate();
end

L = 100;  % Number of iterations
S =100;  %Size of the target image	[S x S]	
target = 255*(randi([0,1], S));

batchInput = zeros(1920, 1080, L);

for ii = 1:L
	input = zeros(1920,1080);
	x = randi([300,1668],1); % I got these limit numbers from Octypus. They are also in the Json file so can be set dynamically if required. 
	y = randi([642,1574],1); %
	batchInput(1:100,1:100,ii) = target;
%	PAD = circshift(PAD, [floor(SI(1)/2)-floor(S(1)/2), floor(SI(2)/2)-floor(S(2)/2)]);
	batchInput(:,:,ii) = circshift(batchInput(:,:,ii), [x,y]);
	
	if (plotStuff)
	figure(1)
	cla()
	imagesc(batchInput(:,:,ii));
	pause(0.01)
	end
end

	% Create fft of target
	padded_target = zeros(1000,1000);
	padded_target(1: size(target,1), 1:size(target,2)) = target; 
	imagesc(padded_target);
	padded_target = circshift(padded_target, [floor(size(padded_target,1)/2 - S/2 ), floor(size(padded_target,2)/2 - S/2)]);
	phase_target = exp(1i * pi * padded_target / 255);
	filter = FT(phase_target);
	phaseFilter = -255.*angle(filter);

	
	% Format the input and the filter batches with the SLM
	% tranformations, as specified in the JSON file. 
	% formatForSLM loops through the number of inputs or filters given to
	% it, and applies the transformations. The string 'input' or 'filter'
	% makes sure the correct SLM transformation from the Json file is
	% applied.
	[FormattedInputBatch] = formatForSLM(camera, slm, json_file, batchInput, 'input');
    [FormattedFilterBatch] = formatForSLM(camera, slm, json_file, phaseFilter, 'filter');
		
	
	
	% send batches to the optics.
	octy_start_batch(json_file, FormattedInputBatch, FormattedFilterBatch);

	% We know that we sent 1 filter and 10 input images to the optics. 
	% We therefore have to read 10 images back. 
	% If we had N inputs and M filters, we'd have to read back NxM if we
	% were doing a all-against-all correlation. 

	tic
	for ii = 1:L
		[camera_photo(:,:,ii), input_image_number, filter_image_number] = octy_pull_batch(json_file);
		if (plotStuff)
		figure(2);
		imagesc(camera_photo(:,:,ii),[0, 50]);  daspect([1,1,1]); colormap gray
		
		title(['Input Image : ', num2str(input_image_number), ' . Filter Image : ', num2str(filter_image_number)]);
		pause(0.05);
		end
	end
	toc
end