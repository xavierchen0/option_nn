# Synopsis
We were largely inspired by [Shvimer and Zhu (2024)](https://doi.org/10.1016/j.eswa.2024.123979), who introduced the idea of combining the simplicity of a parametric model, like Black and Scholes (1973) model, with the flexibility of a neural network to dynamically act as a correction term for the model’s misalignment with actual market option prices.

However, since volatility is the primary driver of an option’s value, the constant volatility assumption in the Black (1973) model is a major source of error due to the well-regarded volatility smile observed in the options market. To improve upon Shvimer and Zhu (2024) idea, we wanted to explicitly provide the neural network with information about the actual shape of the volatility surface. We achieved this by adding two parameters tuned from the SABR (Hagan et al., 2002) model, Rho (correlation between S&P 500 and its volatility) and Nu (Volatility of volatility).

By performing our own hyperparameter tuning, we trained two neural networks for both Call and Put options respectively. This process revealed distinct architectural requirements for each option type: Call options relied primarily on network depth to capture hierarchical non-linearities, while Put options depended on network width for greater capacity. Despite these structural differences, we found that regularisation via dropout was universally critical for preventing overfitting in both models.

When we tested our two neural networks on the test dataset, we discovered that our two neural networks outperformed Shvimer and Zhu (2024) neural networks, Black (1976) model and SABR (Hagan et al., 2002) model by a significant margin. Specifically, our extended neural network reduced the MAE by approximately **25%** compared to Shvimer and Zhu (2024) neural network, and over **70%** compared to the traditional parametric models, Black (1976) and SABR (Hagan et al., 2002) models, thereby proving that the added complexity yielded a substantial return in pricing accuracy.

We also investigated the importance of input parameters using SHAP, and identified that SABR Nu played a significant role in improving our extended neural network’s performance, while SABR Rho had a minimal impact on the performance.

**Overall MAE results for all models**:

| Model | MAE |
| :--- | :--- |
| Our Extended Neural Network | 7.25 |
| Shvimer and Zhu (2024) | 9.66 |
| Black Model | 26.94 |
| SABR Model | 29.24 |

**MAE results by moneyness level and option type**:

| Moneyness Level | Option Type | Our Extended Neural Network | Shvimer and Zhu (2024) | Black Model | SABR Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ATM | Call | 8.16 | 9.81 | 26.53 | 30.54 |
| ITM | Call | 7.76 | 27.53 | 103.19 | 103.34 |
| OTM | Call | 5.89 | 9.78 | 12.0 | 9.24 |
| ATM | Put | 7.32 | 8.38 | 24.43 | 30.84 |
| ITM | Put | 4.56 | 4.65 | 16.87 | 14.74 |
| OTM | Put | 26.70 | 33.78 | 89.26 | 103.97 |

# Project Setup

This project leverages `uv` for efficient project management and `marimo` for interactive notebooks.

## uv

`uv` is an extremely fast Python package installer and resolver, written in Rust. It is used here for dependency management and running project scripts.
Learn more: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

## Marimo

`marimo` is a next-generation Python notebook that's reactive, interactive, and shareable. It provides a new way to write and share Python code with rich outputs.
Learn more: [https://marimo.io](https://marimo.io)
