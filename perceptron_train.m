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
w = zeros(5, 1);
w(5) = 10;
alpha = 2;
errors = zeros(10, 1);
for j=1:10
    errorRate = 0;
    for i=1:100
        input_vec = [data(i, :) 1];
        dot_prod = input_vec * w;
        prediction = 0;
        if dot_prod > 0
            prediction = 1;
        end
        if labels(i) ~= prediction
            errorRate = errorRate + 1;
        end
        w = (w' + alpha * (labels(i) - prediction) * input_vec)';
    end
    errors(j, 1) = errorRate;
    message = strcat({'Error rate on epoch #'}, {int2str(j)}, {' : '}, {int2str(errorRate)}, {'%'});
    %disp(message)
end
figure
scatter([1:10], errors);
title('Error rate vs. epoch number');
xlabel('Epoch number');
ylabel('Error rate');

testdatatable = readtable('iris_test');
testdata = table2array(testdatatable(1:29, 1:4));
testl = table2array(testdatatable(1:29, 5));
testlabels = zeros(100, 1);
for i=1:29
    if length(testl{i}) == 15
        testlabels(i) = 1;
    else
        testlabels(i) = 0;
    end
end

correct = 0;
for i=1:size(testdata, 1)
    input_vec = [testdata(i, :) 1];
    dot_prod = input_vec * w;
    prediction = 0;
    if dot_prod > 0
        prediction = 1;
    end
    if testlabels(i) == prediction
        correct = correct + 1;
    end
end
message = strcat({'Test correct rate: '}, {num2str(correct / 29 * 100)}, {'%'});
disp(message);


