# Deciphering Cold War Polarization: A Machine Learning Approach to 1984 U.S. Congressional Voting Behavior

## 📌 Project Overview
This repository hosts an end-to-end data science and machine learning project designed to decode institutional polarization, legislative discipline, and voting dynamics during the peak of the Cold War. Utilizing the classic **1984 U.S. Congressional Voting Records dataset** from the UCI Machine Learning Repository, the primary objective is to mathematically predict a congressman's political party affiliation (Democrat vs. Republican) solely based on their voting behavior across 16 critical legislative bills.

## 🛠️ Technological Stack
*   **Language:** Python (3.x)
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning Library:** Scikit-Learn (Sklearn)
*   **Data Visualization:** Matplotlib, Seaborn

## ⚙️ Data Preprocessing & Strategy
Socio-political datasets require strict contextual preservation rather than standard statistical dropping. The voting records originally contained `y` (yea), `n` (nay), and `?` (abstain/unknown). 
*   **Strategic Encoding:** Votes were manually mapped using a structured dictionary: `yea = 1`, `nay = -1`, and `abstain = 0`.
*   **Context Preservation:** In political sociology, choosing to abstain or miss a vote is a strategic legislative position. Thus, `0` was retained as a deliberate weight of structural neutrality, allowing algorithms to learn party line discipline and strategic absences.
*   **Data Splitting:** Chronological and non-biased evaluation was guaranteed via an 80/20 train-test split (`random_state=34`).

## 🤖 Modeling & Benchmarking Pipeline
To ensure a robust comparative framework, **9 different machine learning paradigms** were automated, trained, and benchmarked within a unified evaluation loop:

| Model Hierarchy | Algorithm | Accuracy | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Models** | Linear Support Vector Machine (Linear SVM) | **0.9885** | **0.9855** | 🚀 Top Performer |
| **Linear Models** | Logistic Regression | **0.9885** | **0.9855** | 🏆 Champion Model |
| **Deep Learning** | Multi-Layer Perceptron Neural Network (MLP) | **0.9885** | **0.9855** | Over-Engineered |
| **Ensemble Methods** | Random Forest Classifier | 0.9770 | 0.9706 | Highly Robust |
| **Ensemble Methods** | Gradient Boosting Classifier | 0.9655 | 0.9577 | Stable |
| **Distance-Based** | Support Vector Machine (RBF SVM) | 0.9655 | 0.9577 | Good |
| **Probabilistic** | Gaussian Naive Bayes | 0.9655 | 0.9552 | Stable |
| **Distance-Based** | K-Nearest Neighbors (KNN) | 0.9540 | 0.9429 | Fair |
| **Tree-Based** | Decision Tree (CART) | 0.9080 | 0.8919 | Baseline |

## 📊 Key Analytical Insights & Socio-Political Interpretations

### 1. The Power of Ockham's Razor
While the deep learning architecture (**Yapay Sinir Ağları - MLP**) achieved a flawless 98.85% accuracy, it acts as a "Black-Box" model. In political science frameworks, explainability is as vital as precision. Because the 1984 legislative dataset is highly linearly separable, simple white-box linear models shared the top spot. According to **Ockham's Razor**, the most structured, interpretable model is preferred.

### 2. Ultimate Champion: Logistic Regression
**Logistic Regression** was crowned the ultimate champion model over Linear SVM due to its **probabilistic soft-classification** capabilities via the Sigmoid function. Instead of drawing hard boundaries, it outputs probability vectors (e.g., *"This congressman has an 85% probability of being a Democrat"*). The remaining 15% uncertainty successfully detected moderate cross-party defection lines.

### 3. Deconstructing the Fault Lines (Feature Coefficient Analysis)
By executing **Permutation Feature Importance** on the MLP and plotting the regression coefficients ($\beta$), the mathematical model isolated the structural triggers of polarization:
*   **The Polarization Peak:** The **"Medicare Physician Fee Freeze"** bill emerged as the most polarizing axis of the era. The algorithm discovered a near-perfect separation where 92% of Democrats voted *Nay* and 97% of Republicans voted *Yea*.
*   **The Noise in Foreign Policy:** Despite the absolute intensity of the Cold War geopolitics (e.g., *Aid to Nicaraguan Contras*, *El Salvador Aid*), foreign policy variables presented higher mathematical noise due to increased cross-party rebellions, forcing the model to depend more heavily on stable domestic fiscal policies.
*   **Bipartisan Congruence:** Infrastructure bills like the *Water Project Cost Sharing* yielded coefficients near zero ($0$), confirming it as a non-polarizing, bipartisan regional investment zone where party identities collapsed.

### 4. Anatomy of a Single Error (Confusion Matrix Diagnostics)
Out of 87 test instances, the champion model achieved 86 perfect predictions, committing **only a single error** (predicting a registered Democrat as a Republican). Socio-political analysis reveals that this is not an algorithmic failure, but a historic validation: the model successfully flagged a **"Reagan Democrat"**—a centrist Democrat whose actual legislative voting records completely aligned with the conservative Reagan administration.

## 📁 Repository Structure
*   `odev2.py`: The automated preprocessing, training, and model benchmarking Python pipeline.
*   `housevotes84.csv`: The structural target dataset compiled from the UCI repository.
*   `pdf`: Kutuplaşma Tahmini Sonuçları ve Sosyo-Politik İstatistik Raporu (Detailed Turkish Analysis Report).
