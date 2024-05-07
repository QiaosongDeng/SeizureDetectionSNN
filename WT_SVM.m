clc;
clear;
opts = statset('UseParallel', true);

% Define the directory path
folderPath = 'chb-mit-scalp-eeg-database-1.0.0\chb01'; 
%folderPath = 'chb-mit-scalp-eeg-database-1.0.0'; 
allFiles = dir(folderPath);
disp('All files in the directory:');
disp({allFiles.name});

% Initialize a variable to store combined output
label_info = [];

% Get a list of all summary.txt files in the directory
txtfiles = dir(fullfile(folderPath, '*summary.txt'));

% Iterate through each file
for i = 1:length(txtfiles)
    % Construct the full file path
    filePath = fullfile(folderPath, txtfiles(i).name);
    
    % Load or process the data from the file
    % Assuming generate_data_structure() takes a file path as input and returns data
    data = generate_data_structure(filePath);
    
    % Combine the data
    % This assumes that combining data means concatenating it; adjust this logic as needed
    label_info = [label_info; data];
end

%% 


% Get a list of all edf files in the folder that match the pattern
edfFiles = dir(fullfile(folderPath, '*.edf'));

% Initialize an empty array or cell array to store the results
data_with_labels = []; 

% Loop over each edf file and process it
for k = 1:length(edfFiles)
    % Construct the full file path
    edfFilePath = fullfile(folderPath, edfFiles(k).name);

    % Call your function
    result = add_seizure_labels(edfFilePath,label_info);

    % Concatenate the result
    data_with_labels = [data_with_labels; result]; 
end

%% 

% Assuming 'eegData' is your timetable with 23 EEG signal columns and 1 label column
labels = data_with_labels{:, end};  % Extract labels
unique_labels = unique(labels);
disp(unique_labels);
eegSignals = data_with_labels{:, 1:end-1};  % Extract EEG signals

% Parameters
waveletName = 'db4';  % Wavelet type, you can change it
decompositionLevel = 5;  % Level of wavelet decomposition

% Preallocate feature matrix
numFeatures = size(eegSignals, 2) * decompositionLevel;
features = zeros(size(eegSignals, 1), numFeatures);

% Wavelet transform and feature extraction
for i = 1:size(eegSignals, 2)
    for j = 1:size(eegSignals, 1)
        fprintf('%d, %d\n', j, i);
        [C, L] = wavedec(eegSignals{j, i}, decompositionLevel, waveletName);
        % Extracting Mean Absolute Value (MAV) for each level
        for k = 1:decompositionLevel
            coef = appcoef(C, L, waveletName, k);
            featureIndex = (i - 1) * decompositionLevel + k;
            features(j, featureIndex) = mean(abs(coef));  % MAV calculation
        end
    end
end

%% 

fprintf('time to train\n');

% 分割数据集
cv = cvpartition(size(features, 1), 'HoldOut', 0.3);
idx = cv.test;

% 分割训练集和测试集
XTrain = features(~idx, :);
YTrain = labels(~idx, :);
XTest = features(idx, :);
YTest = labels(idx, :);

% 训练SVM模型
%SVMModel = fitcsvm(XTrain, YTrain, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto');
SVMModel = fitcsvm(XTrain, YTrain, 'Standardize', true, 'KernelFunction', 'linear', 'KernelScale', 'auto');

% 测试集预测
[YPredict, score] = predict(SVMModel, XTest);

% 性能评估
confMat = confusionmat(YTest, YPredict);
accuracy = sum(diag(confMat)) / sum(confMat, 'all');
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
f1Score = 2 * (precision * recall) / (precision + recall);

% 计算ROC曲线和AUC
[X, Y, ~, AUC] = perfcurve(YTest, score(:,2), 1);

% 绘制ROC曲线
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve, AUC = ' num2str(AUC)]);

% 打印性能指标结果
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1Score);
fprintf('AUC: %.4f\n', AUC);



function data_with_seizure = add_seizure_labels(edf_filename, label_info)
    % Read the EDF file
    [eeg_data, ~] = edfread(edf_filename);
    disp(edf_filename);
     % 提取文件名和扩展名
    [~, name, ext] = fileparts(edf_filename);
    filenameOnly = [name ext];

    % Initialize the 'Seizure' column with zeros
    seizure_column = zeros(height(eeg_data), 1);

    
    % Find the label_info entry for this EDF file
    label_entry = label_info(strcmp({label_info.File_Name}, filenameOnly));
    
    % If there's no corresponding label_info, return the data as is
    if isempty(label_entry)
        eeg_data.label = seizure_column;
        data_with_seizure = eeg_data;
        return;
    end
    
    % Get the seizure start and end times
    seizure_start_times = [label_entry.Seizure_Start_Time];
    seizure_end_times = [label_entry.Seizure_End_Time];
    
    % Iterate over all seizure periods and mark the 'Seizure' column
    for i = 1:length(seizure_column)
        if i >= seizure_start_times & i <= seizure_end_times
            seizure_column(i+1) = 1;
            fprintf('%s %d second [1]\n',edf_filename,i);
        end
    end
    
    % Add the 'Seizure' column to the timetable
    eeg_data.label = seizure_column;
    
    % Return the modified data
    fprintf('%s labels detected and added\n',edf_filename);
    data_with_seizure = eeg_data;
end



function data_struct = generate_data_structure(filename)
    % Read the file content
    fileID = fopen(filename, 'r');
    file_content = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    file_content = file_content{1};

    % Define patterns
    file_name_pattern = 'File Name: (?<file_name>.*\.edf)';
    file_start_pattern = 'File Start Time: (?<start_time>\d+:\d+:\d+)';
    file_end_pattern = 'File End Time: (?<end_time>\d+:\d+:\d+)';
    seizure_count_pattern = 'Number of Seizures in File: (?<seizure_count>\d+)';
    seizure_start_pattern = 'Seizure Start Time: (?<seizure_start>\d+) seconds';
    seizure_end_pattern = 'Seizure End Time: (?<seizure_end>\d+) seconds';

    % Initialize the data structure
    data_struct = struct('File_Name', {}, 'File_Start_Time', {}, 'File_End_Time', {}, ...
                         'Number_of_Seizures_in_File', {}, 'Seizure_Start_Time', {}, 'Seizure_End_Time', {});

    % Temporary variables to hold current file data
    current_file_name = '';
    current_start_time = '';
    current_end_time = '';
    current_seizure_count = 0;
    current_seizure_start = [];
    current_seizure_end = [];

    % Process the file line by line
    for i = 1:length(file_content)
        line = file_content{i};
        if isempty(line)
            % Add the current file data to the data structure if file name is not empty
            if ~isempty(current_file_name)
                data_struct(end + 1) = struct('File_Name', current_file_name, 'File_Start_Time', current_start_time, ...
                                               'File_End_Time', current_end_time, 'Number_of_Seizures_in_File', current_seizure_count, ...
                                               'Seizure_Start_Time', {current_seizure_start}, 'Seizure_End_Time', {current_seizure_end});
            end
            % Reset temporary variables
            current_file_name = '';
            current_start_time = '';
            current_end_time = '';
            current_seizure_count = 0;
            current_seizure_start = [];
            current_seizure_end = [];
            continue;
        end

        % Check for file name, start time, end time, seizure count, seizure start and end times
        [token, match] = regexp(line, file_name_pattern, 'names', 'match');
        if ~isempty(match)
            current_file_name = token.file_name;
            continue;
        end
        [token, match] = regexp(line, file_start_pattern, 'names', 'match');
        if ~isempty(match)
            current_start_time = token.start_time;
            continue;
        end
        [token, match] = regexp(line, file_end_pattern, 'names', 'match');
        if ~isempty(match)
            current_end_time = token.end_time;
            continue;
        end
        [token, match] = regexp(line, seizure_count_pattern, 'names', 'match');
        if ~isempty(match)
            current_seizure_count = str2num(token.seizure_count);
            continue;
        end
        [token, match] = regexp(line, seizure_start_pattern, 'names', 'match');
        if ~isempty(match)
            current_seizure_start(end + 1) = str2num(token.seizure_start);
            continue;
        end
        [token, match] = regexp(line, seizure_end_pattern, 'names', 'match');
        if ~isempty(match)
            current_seizure_end(end + 1) = str2num(token.seizure_end);
            continue;
        end
    end

    % Add the last entry if it's not empty
    if ~isempty(current_file_name)
        data_struct(end + 1) = struct('File_Name', current_file_name, 'File_Start_Time', current_start_time, ...
                                       'File_End_Time', current_end_time, 'Number_of_Seizures_in_File', current_seizure_count, ...
                                       'Seizure_Start_Time', {current_seizure_start}, 'Seizure_End_Time', {current_seizure_end});
    end
end

function fileList = getAllFiles(dirName, filePattern)
    % 初始化一个空的 fileList
    fileList = {};
    
    % 获取目录下所有文件和文件夹
    dirInfo = dir(dirName);
    
    % 过滤掉 '.' 和 '..'
    dirInfo = dirInfo(~ismember({dirInfo.name}, {'.', '..'}));
    
    % 递归搜索每个子目录
    for i = 1:length(dirInfo)
        if dirInfo(i).isdir
            % 递归调用以搜索子目录
            nextDir = fullfile(dirName, dirInfo(i).name);
            fileList = [fileList; getAllFiles(nextDir, filePattern)];
        else
            % 检查文件是否匹配给定的模式
            if endsWith(dirInfo(i).name, filePattern)
                fileList = [fileList; fullfile(dirName, dirInfo(i).name)];
            end
        end
    end
end
