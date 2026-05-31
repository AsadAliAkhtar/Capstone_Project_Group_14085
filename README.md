# In Search of Alpha in Macroeconomic Hidden Regimes

## Tactical Asset Allocation with HMM and MSVAR

This repository contains the implementation for our MScFE 690 capstone project. The study examines whether hidden macroeconomic regimes can be identified from monthly macroeconomic information and used to improve tactical asset allocation across equities, bonds, and gold.

The central empirical comparison is between:

* Hidden Markov Model (HMM)
* Markov-Switching Vector Autoregression (MSVAR)

Both models are implemented within the same empirical framework so that differences in performance can be interpreted primarily as differences in regime-identification methodology rather than differences in downstream portfolio construction or backtesting assumptions.

## Project Scope

The project follows a monthly macro-driven tactical asset allocation framework.

Macroeconomic inputs are used to infer hidden economic regimes. The resulting regime information is translated into allocation signals across three asset sleeves:

* U.S. equities
* Long-term U.S. Treasury bonds
* Gold

The model-based strategies are evaluated against three benchmark allocations under the same backtesting environment:

* 60/40 equity-bond portfolio
* Equal-weight equity-bond-gold portfolio
* Buy-and-hold equity portfolio

## Repository Structure

```text
Capstone_Project_Group_14085/
├── README.md
├── CONTRIBUTORS.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config/
│   ├── base.yaml
│   └── README.md
├── data/
│   ├── README.md
│   ├── macro_data.csv
│   ├── macro_clean.csv
│   ├── market_data.csv
│   └── market_clean.csv
├── notebooks/
│   ├── README.md
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_wrangling.ipynb
│   ├── 03_hmm_experiments.ipynb
│   ├── 04_msvar_experiments.ipynb
│   └── 05_backtesting.ipynb
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── hmm_model.py
│   ├── msvar.py
│   ├── msvar_model.py
│   ├── portfolio.py
│   ├── regime_mapping.py
│   ├── backtest.py
│   ├── benchmarks.py
│   ├── metrics.py
│   └── plots.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
└── tests/
```

## Implementation Workflow

The repository separates reusable functions from execution notebooks:

* `src/` contains reusable Python functions.
* `notebooks/` contains ordered exploratory and reporting workflows.
* `outputs/` contains generated tables and figures used in the report.

The workflow is:

1. **Data extraction**
   Load macroeconomic series from FRED and asset data from market sources.

2. **Data wrangling**
   Align all series to a monthly month-end index and create model-ready datasets.

3. **HMM estimation**
   Estimate hidden regimes using the Gaussian HMM and export target weights.

4. **MSVAR estimation**
   Estimate hidden regimes using the MSVAR model and export target weights.

5. **Backtesting**
   Compare HMM, MSVAR, and benchmark strategies over the common test period, net of transaction costs.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

To rerun data extraction, set a FRED API key as an environment variable:

```bash
export FRED_API_KEY=your_fred_api_key_here
```

On Windows PowerShell:

```powershell
$env:FRED_API_KEY="your_fred_api_key_here"
```

Then run notebooks in order:

1. `notebooks/01_data_extraction.ipynb`
2. `notebooks/02_data_wrangling.ipynb`
3. `notebooks/03_hmm_experiments.ipynb`
4. `notebooks/04_msvar_experiments.ipynb`
5. `notebooks/05_backtesting.ipynb`

## Main Outputs

Key output tables are saved in `outputs/tables/`, including:

* `hmm_target_weights.csv`
* `msvar_target_weights.csv`
* `net_after_cost_strategy_comparison.csv`
* `net_after_cost_strategy_ranking.csv`
* `subperiod_performance.csv`
* `msvar_vs_hmm_subperiod_advantage.csv`
* `implementation_cost_summary.csv`

Key figures are saved in `outputs/figures/`, including:

* `net_after_cost_strategy_comparison.png`
* `msvar_minus_hmm_sharpe_by_subperiod.png`

## Notes

The notebooks provide the execution narrative and reporting workflow. Reusable model, portfolio, backtest, metric, and plotting functions are implemented in `src/`.
