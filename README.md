# qsar-biodegradability-ml
# 🧪 Predicting Biodegradability of Chemicals using QSAR Data

This project applies machine learning to **classify chemicals as biodegradable or non-biodegradable** using a QSAR dataset of molecular features. It demonstrates a full pipeline including **data preprocessing, model training, feature selection**, and **performance evaluation**.

---

## 📊 Project Summary

- **Dataset**: 1055 chemical samples, 41 features + 1 target (biodegradable = 1, non-biodegradable = 0)
- **Goal**: Accurately predict chemical biodegradability using supervised machine learning
- **Models Used**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM with RBF kernel)
- **Best Accuracy**: 🎯 **90.05% (Random Forest)**

---

## 🔧 Technologies & Methods

- **Programming**: MATLAB
- **Models**: Logistic Regression, Random Forest, SVM
- **Preprocessing**:
  - Missing value check
  - Z-score normalization
  - Train/Test split (80/20)
- **Feature Selection**:
  - Used Random Forest's OOB Permuted Predictor Importance
  - Top 10 features selected for simplified model
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**:
  - Heatmaps of features
  - Performance bar charts
  - Feature importance plots

---

## 📈 Model Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 88.15% | 87.93%    | 73.91% | 80.32%   |
| Random Forest       | **90.05%** | **94.44%**    | **73.91%** | **82.93%**   |
| SVM (RBF Kernel)    | 76.30% | 95.24%    | 28.99% | 44.44%   |
| RF (Top 10 Features)| 88.15% | 88.15%    | 71.01% | 79.68%   |

---

## 🎯 Key Takeaways

- **Random Forest** outperformed other models with the best balance of precision and recall.
- **SVM**, while highly precise, suffered from very low recall and imbalanced predictions.
- **Feature selection** reduced model complexity without significantly sacrificing accuracy.
- This approach shows the **importance of interpretability**, robustness, and simplification in real-world ML applications.

---

## 🧠 Recommendations

- Use Random Forest as the primary model for future deployment.
- Explore **XGBoost**, **Gradient Boosting**, or **LightGBM** for further performance gains.
- Consider **SHAP** or **LIME** for local model interpretability in deployment scenarios.
- Address class imbalance using **SMOTE** or resampling strategies.

---

## 🔮 Future Work

- Extend modeling using **stacked ensembles**.
- Add **feature engineering** and PCA to capture deeper patterns.
- Create a **GUI or web interface** for real-time predictions.

---

## ✍️ Author

**Animesh Sandhu**  
📍 Delhi, India  
📧 Animeshsandhu75@gmail.com  

---

## 📄 References

- Mansouri et al., "QSAR models for ready biodegradability of chemicals," *Journal of Chemical Information and Modeling*, 2013.
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
- Breiman, "Random Forests," *Machine Learning*, 2001.
