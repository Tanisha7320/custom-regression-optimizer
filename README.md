# Machine-learning
# Loss Landscape & Gradient Descent Analysis for Least Squares

A first-principles implementation and spectral analysis of gradient descent optimization applied to least squares regression. This project investigates how design matrix conditioning, eigenvalue bounds, and feature scaling directly dictate optimization convergence rates—moving past black-box library wrappers to analyze the raw linear algebra under the hood.

---

## 📌 Motivation & Core Insight
Modern machine learning heavily relies on high-level libraries (`sklearn`, PyTorch), often treating optimization as a black box. The core motivation behind this project was to understand whats happening behind the libraries:
* **The Math-to-Code Pipeline:** Prove how analyzing loss landscapes, spectral properties, and conditioning turns hyperparameter tuning from random guesswork into a mathematically grounded science.
* **Performance Impact:** Demonstrate that understanding the underlying optimization geometry yields massive conceptual and computational clarity, showing that looking past black-box models can lead to massive computational speed-ups.

---

## 🛠️ The Troubleshooting Journal
Building this from scratch came with critical practical lessons and edge cases:
1. **The Target Normalization Trap:** Encountered deceptively low, unrealistic Mean Squared Error (MSE) early on because the target variable ($y$) was accidentally normalized alongside the features, fundamentally altering the scale of the optimization target.
2. **The Missing Intercept:** Diagnosed massive, stubborn MSEs in the thousands because omitting an intercept term forced the model through the origin, severely misspecifying the line of best fit.
3. **Convexity & Regularization:** Realized that real-world or sparse datasets do not always come with clean strong-convexity bounds ($\mu$-strong convexity), necessitating tools like **Ridge regularization ($L2$)** to guarantee numerical stability and control the condition number.

---

## 📊 Key Results & Spectral Analysis
* **Conditioning Diagnostics:** Conducted spectral analysis on the design matrix Hessian, diagnosing severe ill-conditioning ($\kappa \approx 5.88 \times 10^{10}$). Applying feature normalization reduced $\kappa$ down to $20.88$ (a $2.8 \times 10^9$ reduction factor).
* **Analytical Step-Size Optimization:** Derived optimal step-sizes using $L$-smoothness and $\mu$-strong convexity bounds extracted directly from the Hessian's eigenvalues, accelerating gradient descent convergence by **$28\times$** (dropping from 5,000 to 176 iterations) compared to a baseline random step-size.

---

## 🔍 Scope & Scaling Realities
* **When Exact Math Works:** For low-to-moderate dimensional datasets, computing the exact Hessian and extracting its eigenvalues via spectral analysis ($O(d^3)$) is entirely feasible and offers deep analytical visibility.
* **The Big Data Bottleneck:** At massive industrial scales ($d$ in the millions), computing explicit Hessians or matrix inversions becomes intractable, shifting the architectural necessity toward stochastic approximations (SGD) and iterative matrix-free solvers.
