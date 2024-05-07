clc;
clear;
opts = statset('UseParallel', true);

% Define the directory path
folderPath = 'Dataset'; 

txtfiles = getAllFilesWithPattern(folderPath, 'summary.txt');
label_info = [];

for i = 1:length(txtfiles)
    data = generate_data_structure(txtfiles{i});
    label_info = [label_info, data]; 
end

%% 


% Get a list of all edf files in the folder that match the pattern
edfFiles = getAllFilesWithPattern(folderPath, '.edf');

% Initialize an empty array or cell array to store the results
data_with_labels = []; 

% Loop over each edf file and process it
for k = 1:length(edfFiles)
    % Call your function
    disp(edfFiles{k});
    result = add_seizure_labels(edfFiles{k},label_info);

    % Concatenate the result
    if k == 1
        data_with_labels = result;
        % break; %use this for only geting the first edf file !!!!!!!!!!!!!!!!!!!!!!!!!!!
        continue;
    end

    if ~isequal(width(data_with_labels), width(result))
        disp('两个timetable的尺寸不同。');
    end

    commonColumns = intersect(data_with_labels.Properties.VariableNames, result.Properties.VariableNames);
    data_with_labels = [data_with_labels(:, commonColumns); result(:, commonColumns)];

end

%% 

% Assuming 'eegData' is a timetable with 23 EEG signal columns and 1 label column
labels = data_with_labels{:, end};  % Extract labels
unique_labels = unique(labels);
disp(unique_labels);
eegSignals = data_with_labels{:, 1:end-1};  % Extract EEG signals

% Save labels into csv file
% % Expand each element to 256 rows
% expandedLabels = repmat(labels, 1, 256);
% % Reshape the matrix to a single column
% finalLabels = reshape(expandedLabels', [], 1);
% writematrix(finalLabels, 'labels.csv');

writematrix(labels, 'labels.csv');

%% 

eegSignalsFiltered = cell(size(eegSignals));  % 创建一个数组来存储过滤后的 EEG 信号

samplingRate = 256;
% 应用带阻滤波器以去除电源线噪声
d = designfilt('bandstopiir', 'FilterOrder', 2, ...
               'HalfPowerFrequency1', 59, 'HalfPowerFrequency2', 61, ...
               'DesignMethod', 'butter', 'SampleRate', samplingRate);

% 创建两个带通滤波器
bpFilt1 = designfilt('bandpassiir', 'FilterOrder', 20, ...
            'HalfPowerFrequency1', 3, 'HalfPowerFrequency2', 8, ...
            'SampleRate', samplingRate);
bpFilt2 = designfilt('bandpassiir', 'FilterOrder', 20, ...
            'HalfPowerFrequency1', 8, 'HalfPowerFrequency2', 16, ...
            'SampleRate', samplingRate);

for i = 1:size(eegSignals, 2)
    for j = 1:size(eegSignals, 1)
        fprintf('%d, %d\n', j, i);
        
        % 应用带阻滤波器
        eegDataFiltered = filtfilt(d, eegSignals{j, i});

        % 应用两个带通滤波器
        eegDataBand1 = filtfilt(bpFilt1, eegDataFiltered);
        eegDataBand2 = filtfilt(bpFilt2, eegDataFiltered);

        % 存储过滤后的数据
        eegSignalsFiltered{j, i, 1} = eegDataBand1;  % 3-8 Hz
        eegSignalsFiltered{j, i, 2} = eegDataBand2;  % 8-16 Hz
    end
end

% 预分配特征矩阵
numFeatures = size(eegSignalsFiltered, 2) * 2;  % 每个信号两个特征：两个频率带的 MAV
features = zeros(size(eegSignalsFiltered, 1), numFeatures);

% 从过滤后的信号中提取特征
for i = 1:size(eegSignalsFiltered, 2)
    for j = 1:size(eegSignalsFiltered, 1)
        % 计算并存储 3-8 Hz 和 8-16 Hz 频率带的 MAV
        features(j, (i - 1) * 2 + 1) = mean(abs(eegSignalsFiltered{j, i, 1}));  % 3-8 Hz MAV
        features(j, (i - 1) * 2 + 2) = mean(abs(eegSignalsFiltered{j, i, 2}));  % 8-16 Hz MAV
    end
end

writematrix(features, 'features.csv');





%% 

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
    
    if length(label_entry)>1
        disp("!!");
    end

    % Get the seizure start and end times
    seizure_start_times = [label_entry.Seizure_Start_Time];
    seizure_end_times = [label_entry.Seizure_End_Time];
    
    % Iterate over all seizure periods and mark the 'Seizure' column
    for j = 1:length(seizure_start_times)
        for i = 1:length(seizure_column)
            if i >= seizure_start_times(j) & i <= seizure_end_times(j)
                seizure_column(i+1) = 1;
                fprintf('%s %d second [1]\n',edf_filename,i);
            end
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
    % seizureS_start_pattern = 'Seizure \d+ Start Time: (?<seizure_start>\d+) seconds';
    seizureS_start_pattern = 'Seizure \d+ Start Time: \s*(?<seizure_start>\d+) seconds';
    % seizureS_end_pattern = 'Seizure \d+ End Time: (?<seizure_end>\d+) seconds';
    seizureS_end_pattern = 'Seizure \d+ End Time: \s*(?<seizure_end>\d+) seconds';


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

        if current_seizure_count > 1 && ~isempty(current_seizure_end)
            if ~isempty(current_file_name)
                data_struct(end + 1) = struct('File_Name', current_file_name, 'File_Start_Time', current_start_time, ...
                                               'File_End_Time', current_end_time, 'Number_of_Seizures_in_File', current_seizure_count, ...
                                               'Seizure_Start_Time', {current_seizure_start}, 'Seizure_End_Time', {current_seizure_end});
            end
            current_seizure_start = [];
            current_seizure_end = [];
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

        [token, match] = regexp(line, seizureS_start_pattern, 'names', 'match');
        if ~isempty(match)
            current_seizure_start(end + 1) = str2num(token.seizure_start);
            continue;
        end
        [token, match] = regexp(line, seizureS_end_pattern, 'names', 'match');
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

% 创建一个函数来递归遍历文件夹
function fileList = getAllFilesWithPattern(dirPath, pattern)
    % 获取文件夹中的所有内容
    dirContent = dir(dirPath);
    % 过滤掉'.'和'..'
    dirContent = dirContent(~ismember({dirContent.name}, {'.', '..'}));
    
    % 初始化文件列表
    fileList = {};
    
    % 遍历文件夹内容
    for i = 1:length(dirContent)
        % 获取完整路径
        currentPath = fullfile(dirPath, dirContent(i).name);
        
        % 检查是否是文件夹
        if dirContent(i).isdir
            % 如果是文件夹，则递归调用
            fileList = [fileList; getAllFilesWithPattern(currentPath, pattern)];
        else
            % 检查文件名是否符合指定模式
            if endsWith(dirContent(i).name, pattern)
                % 如果是符合条件的文件，则添加到列表中
                fileList = [fileList; {currentPath}];
            end
        end
    end
end