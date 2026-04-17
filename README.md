# Capstone_Project_Group_14085
This repository serves as a contribution and collaboration platform for the capstone project

# MScFE 690 Capstone Project
## In Search of Alpha in Macroeconomic Hidden Regimes: A Tactical Asset Allocation Framework

This repository contains the implementation for our MScFE 690 capstone project.  
The study examines whether hidden macroeconomic regimes can be identified from observable monthly macroeconomic information and used to improve tactical asset allocation across equity, bond, and gold sleeves.

The central empirical comparison in this project is between:
- **Hidden Markov Model (HMM)**
- **Markov-Switching VAR (MS-VAR)**

Both models are implemented within the same empirical framework so that differences in results can be interpreted as differences in regime-identification methodology rather than differences in downstream portfolio construction or backtesting setup.

---

## Project Scope

The project is organized around a monthly macro-driven tactical asset allocation framework.  
Macroeconomic inputs are used to infer hidden economic regimes, and the resulting regime information is translated into allocation signals for a multi-asset portfolio spanning:

- **Equities**
- **Bonds**
- **Gold**

The model-based strategies are evaluated against benchmark allocations under the same backtesting environment.

---

## Code Plan and Implementation Architecture

The repository is structured as a unified research pipeline with the following stages:

1. **Data ingestion**  
   Load monthly macroeconomic and asset series.

2. **Preprocessing**  
   Align dates, clean data, and transform variables into a model-ready dataset.

3. **Model estimation**  
   Estimate hidden regimes using HMM and MS-VAR on the same processed sample.

4. **Regime extraction**  
   Obtain hidden states and regime probabilities for each model.

5. **Portfolio construction**  
   Convert model-implied regime information into allocation signals across equity, bond, and gold sleeves.

6. **Backtesting**  
   Evaluate the resulting strategies using a common rebalancing framework, benchmark definitions, and transaction cost assumptions.

7. **Performance evaluation**  
   Compare strategies using standard portfolio measures such as return, volatility, Sharpe ratio, and maximum drawdown.

This architecture is intended to preserve consistency across model runs and support a controlled comparison between HMM and MS-VAR within the same tactical allocation problem.

---

## Repository Structure

```text
Capstone_Project_Group_14085/
├── README.md
├── CONTRIBUTORS.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── base.yaml
│   └── README.md
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── hmm_model.py
│   ├── msvar_model.py
│   ├── regime_mapping.py
│   ├── backtest.py
│   ├── benchmarks.py
│   ├── metrics.py
│   └── plots.py
├── notebooks/
│   └── README.md
└── outputs/
    ├── figures/
    │   └── README.md
    ├── tables/
    │   └── README.md
    └── logs/
        └── README.md
