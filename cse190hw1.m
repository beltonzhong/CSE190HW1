% read in data and labels
irisdata = readtable('iris_data');

% separate data from labels
data = irisdata(:, [1:4]);
labels = irisdata(:, 5);

% calculate average and standard dev of each column
avgs = mean(data);
std_devs = std(data);

% calculate z scores
zscore_data = zeros(size(data));
for i=1:5
    zscore_data(:, i) = (data(:, i) - avgs(i)) / std_devs(i);
end




