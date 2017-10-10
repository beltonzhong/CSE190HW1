% read in data and labels
irisdata = readtable('iris_data');

% separate data from labels
data = table2array(irisdata(1:100, 1:4));
l = table2array(irisdata(1:100, 5));
labels = zeros(100, 1);
for i=1:100
    if length(l{i}) == 15
        labels(i) = 1;
    else
        labels(i) = 0;
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
alpha = 0.5;

for j=1:100
    for i=1:100
        input_vec = data(i, :);
        dot_prod = input_vec * w;
        prediction = 0;
        if dot_prod > 0
            prediction = 1;
        end
        w = w + alpha * (labels(i) - prediction) * input_vec;
    end
    display(w)
end

output = zeros(100, 1);

for i=1:100
    input_vec = data(i, :);
    dot_prod = input_vec * w;
    prediction = 0;
    if dot_prod > 0
        prediction = 1;
    end
    output(i) = prediction;
end
