function [camera,slm,json_file] = setupOpticalCorrelate(json_file)
%% setupOpticalCorrelate
% This function runs the octy_config function to establish the camera and
% SLM structures which will be used later

%% THIS IS EFFECTIVELY LIFED FROM test_octy_call_device.m

addpath('/home/optalysys/OptalysysSoftware/matlab/Sources/Matlab/Plugin');
if nargin<1
 	
	json_file = '/home/optalysys/OptalysysSoftware/JsonDriverFactory/Sony_System_BP.json';
	
end
    
[camera, slm] = octy_config(json_file);
end