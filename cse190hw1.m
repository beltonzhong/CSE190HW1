% read in data and labels
irisdata = readtable('iris_data');

% separate data from labels
data = table2array(irisdata(:, [1:4]));
labels = table2array(irisdata(:, 5));

% calculate average and standard dev of each column
avgs = mean(data);
std_devs = std(data);

% calculate z scores
zscore_data = zeros(size(data));
for i=1:4
    zscore_data(:, i) = (data(:, i) - avgs(i)) / std_devs(i);
end

% generate scatter plots

figure

subplot(3, 2, 1);
hold on
scatter(zscore_data([1:50], 1), zscore_data([1:50], 2));
scatter(zscore_data([51:100], 1), zscore_data([51:100], 2), 'filled');
title('Sepal length vs. Sepal width');

subplot(3, 2, 2)
hold on
scatter(zscore_data([1:50], 1), zscore_data([1:50], 3));
scatter(zscore_data([51:100], 1), zscore_data([51:100], 3), 'filled');
title('Sepal length vs. Petal length');

subplot(3, 2, 3)
hold on
scatter(zscore_data([1:50], 1), zscore_data([1:50], 4));
scatter(zscore_data([51:100], 1), zscore_data([51:100], 4), 'filled');
title('Sepal length vs. Petal width');

subplot(3, 2, 4)
hold on
scatter(zscore_data([1:50], 2), zscore_data([1:50], 3))
scatter(zscore_data(51:100, 2), zscore_data(51:100, 3), 'filled');
title('Sepal width vs. Petal length');

subplot(3, 2, 5)
hold on
scatter(zscore_data([1:50], 2), zscore_data([1:50], 4));
scatter(zscore_data([51:100], 2), zscore_data([51:100], 4), 'filled');
title('Sepal width vs. Petal width');

subplot(3, 2, 6)
hold on
scatter(zscore_data([1:50], 3), zscore_data([1:50], 4));
scatter(zscore_data([51:100], 3), zscore_data([51:100], 4), 'filled');
title('Petal length vs. Petal width');

 

