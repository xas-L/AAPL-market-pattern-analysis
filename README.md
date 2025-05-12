# AAPL Stock Market Pattern Analysis & Predictive Modelling

## Project Overview

This project initially explores the historical price movements of Apple Inc. (AAPL) stock using **unsupervised machine learning** techniques. The primary goal of this phase was to automatically identify distinct **market regimes** (periods with characteristic volatility and trend behavior) and detect **anomalous trading days** (days with unusual price or volume activity) within the dataset spanning from December 1980 to mid-2022. This served as an exercise in uncovering hidden patterns within financial time series using only OHLCV data. 

Building upon this exploratory analysis, the project scope extends into **supervised predictive modelling**. The subsequent phase, detailed in the "Future Work and Research Directions" section, outlines a rigorous plan to develop and evaluate statistical and machine learning models for forecasting short-to-medium term AAPL stock returns, potentially integrating insights from textual analysis (NLP) in later stages.

**Data Source:** `AAPL.csv` from Hugging Face (`Ammok/apple_stock_price_from_1980-2021`) or similar source. (Expected columns: Date, Open, High, Low, Close, Adj Close, Volume)

### Original Inception: August 2023 private repository, cloned and shared May 2025

## Completed Analysis: Unsupervised Pattern Detection

This section details the completed unsupervised analysis phase.

### 1. Data Preparation and Feature Engineering
* **Loading:** `AAPL.csv` loaded, dates parsed, indexed by date. Basic cleaning applied. Adjusted Close (`Adj Close`) used.
* **Feature Calculation:** Key time-series features calculated:
    * `log_return`: Daily logarithmic return.
    * `volatility`: Annualized rolling standard deviation of log returns.
    * `momentum`: Annualized rolling mean of log returns.
    * `volume_log_ratio`: Logarithm of the ratio between daily volume and its rolling mean.
* **Window Sizes:** Features engineered using 60-day and 21-day rolling windows.
* **Scaling:** Features standardized using `sklearn.preprocessing.StandardScaler`.

### 2. Market Regime Detection (K-Means)
* Applied to scaled `volatility` and `momentum`.
* Elbow Method used to guide K selection.
* Regimes identified and visualized for K=3, 4, 5, 6 (60-day window) and K=4 (21-day window).

### 3. Anomaly Detection
* **Isolation Forest:** Applied to all four scaled features (60-day and 21-day windows).
* **Local Outlier Factor (LOF):** Applied to all scaled features (60-day window).
* **Z-Score Method:** Applied to scaled `log_return` (threshold=3.5 std dev, 60-day window).
* Anomalies identified and visualized for each method and window size.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # macOS/Linux: python3 -m venv venv && source venv/bin/activate
    # Windows: python -m venv venv && .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Unsupervised Analysis)

1.  Place your `AAPL.csv` file (or equivalent) in a `data/` directory (you might need to create it).
2.  Run the unsupervised analysis script (assuming it's `src/analysis.py`):
    ```bash
    python src/analysis.py
    ```
3.  Console output will show progress.
4.  Generated plots from the unsupervised analysis will be saved to the `results/plots/` directory.

## Future Work and Research Directions

This project provides a foundation for a comprehensive research plan focused on predictive modelling and integrating alternative data sources. The research brief covers some of what will be outlined, the following outlines the planned next steps:

### 1. Supervised Predictive Modelling (Core Research Plan)
* **Objective:** Develop and rigorously evaluate models to predict N-day ahead logarithmic returns ($y_t = \log(P_{t+N}/P_t)$) for AAPL, using historical OHLCV data. Prediction horizons (N) will explore short (1, 5 days) and medium (21 days) terms.
* **Feature Engineering:** Expand feature set beyond the unsupervised phase:
    * *Lagged Variables:* Include lagged returns, volatility, and potentially other normalized OHLCV features (e.g., $P_t / \text{SMA}_k(P_t)$).
    * *Technical Indicators:* Compute standard indicators (RSI, MACD, Bollinger Bands, ATR, ADX, etc.), ensuring no lookahead bias.
    * *Time-Based Features:* Encode calendar effects (day of week/month/year).
    * *(Optional) Unsupervised Features:* Incorporate regime labels or anomaly scores from the initial analysis as potential predictive features.
    * *Feature Selection:* Apply filter, wrapper, or embedded methods (e.g., correlation, RFE, Lasso, Tree Importance) to identify salient predictors.
* **Data Handling:**
    * *Temporal Splitting:* Strict chronological splits into Training, Validation, and Test sets.
    * *Walk-Forward Validation:* Potential use for more robust hyperparameter tuning.
    * *Scaling:* Fit scalers (`StandardScaler`, `MinMaxScaler`) *only* on training data and apply to validation/test sets.
* **Model Selection & Training:** Explore a hierarchy of models:
    * *Baselines:* Historical Mean, Naive/Persistence forecast.
    * *Linear Models:* OLS, Ridge, Lasso.
    * *Non-Linear ML:* SVR (various kernels), Tree Ensembles (Random Forest, XGBoost, LightGBM, CatBoost).
    * *Deep Learning:* LSTMs, GRUs, potentially 1D CNNs and Transformers, adapted for time series.
* **Hyperparameter Tuning:** Use Grid Search, Randomized Search, or Bayesian Optimization on the *validation set*.
* **Rigorous Evaluation:**
    * *Metrics:* MSE, RMSE, MAE, R-squared (out-of-sample), Directional Accuracy (Hit Rate).
    * *Significance Testing:* Use Diebold-Mariano test to compare model forecasts against baselines.
    * *Financial Backtesting (Simulated):* Implement simple strategies based on predictions, calculate performance (Sharpe, Drawdown, Returns), incorporating transaction cost estimates. **Interpret with extreme caution due to overfitting risks.**
* **Addressing Challenges:** Explicitly consider non-stationarity, low signal-to-noise, overfitting, lookahead bias, and the Efficient Market Hypothesis.

### 2. Advanced NLP Integration (Future Expansion)
* **Rationale:** Incorporate information from unstructured textual data (e.g., equity research reports from sources like UBS, JPM) to capture sentiment, forward-looking statements, and qualitative assessments potentially missed by OHLCV data alone.
* **Approach:**
    * *Data Acquisition:* Source and parse historical AAPL equity research reports (significant challenge).
    * *Sentiment Analysis:* Apply NLP models (Lexicon-based, FinBERT, other Transformers) to extract sentiment scores. Explore Aspect-Based Sentiment Analysis.
    * *Feature Generation:* Create numerical features from sentiment scores (e.g., recent average sentiment, changes in sentiment/ratings).
    * *Model Integration:* Add sentiment features to the supervised models and evaluate their marginal contribution via ablation studies and feature importance.

### 3. Reinforcement Learning (Exploratory)
* Explore using RL (e.g., DQN, Actor-Critic) to dynamically weight technical vs. sentiment signals based on market context. This is a highly complex and research-intensive direction.

### 4. Other Unsupervised Extensions
* Explore GMMs or HMMs for regime modelling.
* Utilize Autoencoders for anomaly detection.
* Develop interactive visualizations (Dash/Streamlit).

## Dependencies

Python package requirements are listed in `requirements.txt`. Key dependencies include:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* *(Potentially others for future work: tensorflow/pytorch, statsmodels, specific NLP libraries like transformers, etc.)*

## Disclaimer: Project Origin and Context

This research project was initiated on August 17th, 2023. The foundational work and conceptualization occurred during the author's internship tenure on an equities desk at an established, though unnamed, asset management firm.

The analysis, methodologies, and code presented herein represent the author's original work, and all intellectual property rights associated with this specific research plan and its potential implementation belong solely to the author.

The project initially formed part of a paper trading strategy developed for an internal competition involving interns at the aforementioned asset manager and participants from three unnamed hedge funds based in London. This competition involved tracking performance metrics such as risk-adjusted returns, Sharpe ratio, and maximum drawdown across various self-selected asset classes (the author focused on equities and equity options). Trading strategies and specific positions were not disclosed between participants, High-Frequency Trading (HFT) strategies were disallowed, and standardized assumptions regarding transaction costs and slippage were applied where relevant.

The decision to keep the involved firms and individuals anonymous is deliberate, respecting the privacy and confidentiality of former colleagues and peers.

## License

MIT