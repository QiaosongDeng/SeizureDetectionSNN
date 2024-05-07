clc;
clear;
[eeg_data1, ~] = edfread('chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf'); 
[eeg_data5, ~] = edfread('chb-mit-scalp-eeg-database-1.0.0\chb05\chb05_39.edf'); 
[eeg_data6, ~] = edfread('chb-mit-scalp-eeg-database-1.0.0\chb06\chb06_01.edf');
% excludedColumns = contains(eeg_data.Properties.VariableNames, 'LOC') | contains(eeg_data.Properties.VariableNames, 'ECG') | contains(eeg_data.Properties.VariableNames, 'VNS');


