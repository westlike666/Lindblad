close all
% Define the power-law function to fit the data
powerlaw = @(a, b, x) a * x.^b;

% Sample data (replace with your own data)
x = [1:11]'
;
for i=[1:5]
y = Y(1:end,i)    
% Define lower and upper bounds for the fitting parameters
lower_bounds = [Y(1,i), -inf]; % [a_min, b_min]
upper_bounds = [Y(1,i), Inf]; % [a_max, b_max]

% Fit the data using the power-law function with bounds
[fit_result, gof] = fit(x, y, fittype(powerlaw), 'Lower', lower_bounds, 'Upper', upper_bounds);



% Print the fitted parameters
a = fit_result.a;
b = fit_result.b;
fprintf('Fitted parameters: a=%.4f, b=%.4f\n', a, b);

% Generate the fitted curve
x_fit = linspace(min(x), max(x), 100);
y_fit = powerlaw(a, b, x_fit);

% Plot the original data and the fitted curve
%figure;
scatter(x, y, 'blue', 'DisplayName', 'Original data');
hold on;
plot(x_fit, y_fit, 'red', 'DisplayName', sprintf('Fitted power-law curve: y=%.4fx^{%.4f}', a, b));
xlabel('x');
ylabel('y');
set()
legend;
%hold off;
end