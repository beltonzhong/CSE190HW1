% read in data and labels
irisdata = readtable('iris_data');

% separate data from labels
data = table2array(irisdata(1:100, 1:4));
l = table2array(irisdata(1:100, 5));
labels = zeros(100, 1);
for i=1:100
    display(l{i})
    if l{i} == 'Iris-versicolor'
        labels(i) = 1
    else
        labels(i) = 0
    end
end

% calculate average and standard dev of each column
avgs = mean(data);
std_devs = std(data);

% calculate z scores
zscore_data = zeros(size(data));
for i=1:4
    zscore_data(:, i) = (data(:, i) - avgs(i)) / std_devs(i);
end

% - is label 0: iris-setosa
% + is label 1: iris-versicolor
w = zeros(4, 1);
for i=1:100
    input_vec = data(i, :);
    dot_prod = input_vec * w;
    if (labels(i) == 'Iris-setosa' && sign(dot_prod) ~= 0) || (labels(i) == 'Iris-versicolor' && sign(dot_prod) ~= 1)
        
    end
     
    
end
