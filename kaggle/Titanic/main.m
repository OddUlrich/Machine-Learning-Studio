%% Titanic: Machine Learning from disaster. (Classification)

% Instructions
% ------------
%
% The following functions will be used in this exercise:
%       costFunction
%       predict

%% Initialization
clear; close all; clc;

%% ================== Loading the data ================== 
% Before loading the data, some features that are irrelevant to the
% practical situation of Titanic disaster should be ignored. The rest data
% will store in a Matrix with examples in column and features in rows.

% Load Training Data
fprintf('Loading Data ...\n');

Input = csvread('train.csv');

% Pick up single feature from original data




% Useful values
m = size(X, 1);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Compute Cost and Gradient ================== 
% In this part, original data is used as features with any preoperation.

[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1), X];

% Initialize fitting parameters
theta = zeros(n + 1, 1);

% Computer and display initial cost and gradient
[cost, grad] = costFunction(theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================== Update Theta ================== 
learning_rate = 0.01;

theta = updateParaFunc(theta, grad, learning_rate);

fprintf('Value of updated theta: \n');
fprintf(' %f \n', theta);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================== Predict and Tranform ================== 

p = predict(theta, X);

fprintf(' %d \n', p);
% write into a .csv file
% write();











