% Predicting Biodegradability of Chemicals using QSAR Data 
% ------------------------------------------------------- 
%% 1. Load and Inspect the Dataset 
load('QSAR_data.mat'); 
% Display dataset information 
disp('Dataset size:'); 
disp(size(QSAR_data)); 
% Separate features and labels 
X = QSAR_data(:, 1:41); % Features 
y = QSAR_data(:, 42);   % Labels (1 = Biodegradable, 0 = Non-Biodegradable) 
% Check for missing values 
if any(isnan(QSAR_data), 'all') 
disp('Warning: Missing values found in the dataset.'); 
else 
disp('No missing values in the dataset.'); 
end 
% Plot Heatmap of Features 
f
 igure; 
imagesc(X); % Visualize the raw features 
colorbar; 
title('Heatmap of QSAR Features'); 
xlabel('Feature Index'); 
ylabel('Sample Index'); 
%% 2. Data Preprocessing 
% Normalize the features 
X_mean = mean(X, 1); 
X_std = std(X, 0, 1); 
X_norm = (X - X_mean) ./ X_std; 
% Train-test split 
cv = cvpartition(size(X_norm, 1), 'HoldOut', 0.2); 
X_train = X_norm(cv.training, :); 
y_train = y(cv.training); 
X_test = X_norm(cv.test, :); 
y_test = y(cv.test); 
%% 3. Train Models 
% Logistic Regression 
logistic_model = fitclinear(X_train, y_train, 'Learner', 'logistic'); 
% Random Forest 
rf_model = TreeBagger(100, X_train, y_train, 'OOBPredictorImportance', 'on'); 
% SVM 
svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf'); 
%% 4. Evaluate Models 
% Predictions 
y_pred_logistic = predict(logistic_model, X_test); 
y_pred_rf = str2double(predict(rf_model, X_test)); 
y_pred_svm = predict(svm_model, X_test); 
% Model Evaluation 
disp('--- Model Performance Metrics ---'); 
[acc_logistic, prec_logistic, rec_logistic, f1_logistic] = evaluateModel(y_test, y_pred_logistic, 
'Logistic Regression'); 
[acc_rf, prec_rf, rec_rf, f1_rf] = evaluateModel(y_test, y_pred_rf, 'Random Forest'); 
[acc_svm, prec_svm, rec_svm, f1_svm] = evaluateModel(y_test, y_pred_svm, 'SVM'); 
% Combine Metrics into Table 
ModelNames = {'Logistic Regression', 'Random Forest', 'SVM'}; 
Accuracy = [acc_logistic, acc_rf, acc_svm]; 
Precision = [prec_logistic, prec_rf, prec_svm]; 
Recall = [rec_logistic, rec_rf, rec_svm]; 
F1_Score = [f1_logistic, f1_rf, f1_svm]; 
PerformanceTable = table(ModelNames', Accuracy', Precision', Recall', F1_Score', ... 
'VariableNames', {'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'}); 
disp(PerformanceTable); 
%% 5. Visualize Performance Comparison 
f
 igure; 
bar([Accuracy; Precision; Recall; F1_Score]'); 
set(gca, 'XTickLabel', ModelNames); 
legend('Accuracy', 'Precision', 'Recall', 'F1-Score'); 
ylabel('Metric Value'); 
title('Performance Comparison of Models'); 
%% 6. Feature Importance in Random Forest 
importance = rf_model.OOBPermutedPredictorDeltaError; 
% Plot feature importance 
f
 igure; 
bar(importance); 
xlabel('Feature Index'); 
ylabel('Importance'); 
title('Feature Importance in Random Forest'); 
%% 7. Evaluate Random Forest with Top Features 
% Select Top 10 Features 
[~, top_idx] = sort(importance, 'descend'); 
top_features_idx = top_idx(1:10); 
% Retrain using top features 
X_train_top = X_train(:, top_features_idx); 
X_test_top = X_test(:, top_features_idx); 
rf_model_top = TreeBagger(100, X_train_top, y_train, 'OOBPrediction', 'on'); 
y_pred_rf_top = str2double(predict(rf_model_top, X_test_top)); 
[acc_rf_top, ~, ~, ~] = evaluateModel(y_test, y_pred_rf_top, 'Top Features RF'); 
disp(['Random Forest Accuracy with Top Features: ', num2str(acc_rf_top)]); 
%% 8. Final Accuracy Comparison 
ModelNames_Final = {'Logistic Regression', 'Random Forest', 'Top Features RF'}; 
Accuracies_Final = [acc_logistic, acc_rf, acc_rf_top]; 
% Grouped bar chart 
f
 igure; 
bar(Accuracies_Final); 
set(gca, 'XTickLabel', ModelNames_Final); 
ylabel('Accuracy'); 
title('Final Accuracy Comparison'); 
 
%% Function: Evaluate Model Metrics 
function [accuracy, precision, recall, f1] = evaluateModel(y_true, y_pred, modelName) 
    % Confusion Matrix 
    cm = confusionmat(y_true, y_pred); 
    TN = cm(1,1); FP = cm(1,2); 
    FN = cm(2,1); TP = cm(2,2); 
 
    % Metrics 
    accuracy = (TP + TN) / (TP + TN + FP + FN); 
    precision = TP / (TP + FP); 
    recall = TP / (TP + FN); 
    f1 = 2 * (precision * recall) / (precision + recall); 
 
    % Display Metrics 
    disp(['--- ', modelName, ' ---']); 
    disp(['Accuracy: ', num2str(accuracy)]); 
    disp(['Precision: ', num2str(precision)]); 
    disp(['Recall: ', num2str(recall)]); 
    disp(['F1-Score: ', num2str(f1)]); 
end 
 
 