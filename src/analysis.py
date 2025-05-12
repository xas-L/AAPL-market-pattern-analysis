import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor # For LOF anomaly detection
from sklearn.impute import SimpleImputer
import os # To create directories for results

# --- Configuration Constants ---
# Input data file path
CSV_PATH = "AAPL.csv" # Make sure this points to your data file
# Directory to save plots
PLOTS_DIR = "results/plots" # Plots will be saved here

# Feature Engineering Parameters
ORIGINAL_WINDOW_SIZE = 60
NEW_WINDOW_SIZE = 21

# Regime Detection Parameters
DEFAULT_REGIME_K = 4
REGIME_KS_TO_TRY = [3, 5, 6] # Additional K values for regimes
REGIME_FEATURES = ['volatility', 'momentum'] # Features used for regime clustering

# Anomaly Detection Parameters
ANOMALY_METHODS_TO_TRY = ['IsolationForest', 'LOF', 'ZScore']
LOF_N_NEIGHBORS = 20 # Parameter for LOF
ZSCORE_THRESHOLD = 3.5 # Threshold for Z-score anomalies
ZSCORE_FEATURE = 'log_return' # Feature for Z-score method

# General Parameters
PRICE_COLUMN = 'Adj Close' # Default, will verify later
RANDOM_STATE = 42

# --- Utility Functions ---

def load_and_prep_data(csv_path):
    """
    Loads stock data from a CSV file, parses dates, sets index,
    verifies required columns, and handles basic cleaning.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame or None: The prepared DataFrame, or None if errors occur.
    """
    global PRICE_COLUMN # Allow modification of the global variable

    print(f"--- Loading and Preparing Data from '{csv_path}' ---")
    try:
        # Try reading with default UTF-8, fallback to latin1 if needed
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin1 encoding.")
            df = pd.read_csv(csv_path, encoding='latin1')
        print(f"CSV '{csv_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'. Please ensure the file exists in the specified location.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Date Parsing and Indexing
    try:
        if 'Date' not in df.columns:
            raise KeyError("'Date' column not found. Check CSV header.")
        # Attempt to parse dates, handling potential errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where date parsing failed
        invalid_dates = df['Date'].isnull().sum()
        if invalid_dates > 0:
            print(f"Warning: Dropping {invalid_dates} rows with invalid date formats.")
            df.dropna(subset=['Date'], inplace=True)
        if df.empty:
             print("Error: No valid date entries found after parsing.")
             return None
        df = df.set_index('Date')
        df = df.sort_index()
        print("Date parsing and indexing successful.")
    except KeyError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error processing Date column: {e}")
        return None

    # Verify Price and Volume Columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if PRICE_COLUMN not in df.columns:
        if 'Close' in df.columns:
            PRICE_COLUMN = 'Close' # Fallback to 'Close'
            print(f"Warning: 'Adj Close' not found. Using 'Close'. Results may be affected by splits/dividends.")
        else:
            print(f"Error: Neither 'Adj Close' nor 'Close' found in columns: {list(df.columns)}")
            return None
    required_cols.append(PRICE_COLUMN)

    missing_req_cols = [col for col in required_cols if col not in df.columns]
    if missing_req_cols:
        print(f"Error: Missing required columns: {missing_req_cols}")
        return None

    # Convert relevant columns to numeric, coercing errors
    print("Converting OHLCV columns to numeric...")
    for col in required_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic Cleaning (Impute NaNs, handle potential zeros)
    numeric_cols_to_check = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    cols_to_impute = [col for col in numeric_cols_to_check if col in df.columns and df[col].isnull().any()]
    if cols_to_impute:
        print(f"Imputing NaNs in {cols_to_impute} using ffill then bfill.")
        df[cols_to_impute] = df[cols_to_impute].ffill().bfill()
        # Check if any NaNs remain after imputation (e.g., if entire column was NaN)
        if df.isnull().values.any():
             print("Warning: NaNs remain after imputation. Dropping rows with any remaining NaNs.")
             df.dropna(inplace=True)

    # Handle potential zero/negative values before log/division operations
    if (df[PRICE_COLUMN] <= 0).any():
         print(f"Warning: Zero or negative values found in '{PRICE_COLUMN}'. Clipping to a small positive value (1e-6).")
         df[PRICE_COLUMN] = df[PRICE_COLUMN].clip(lower=1e-6)
    if 'Volume' in df.columns and (df['Volume'] <= 0).any():
         print("Warning: Zero or negative values found in 'Volume'. Clipping to 1.")
         df['Volume'] = df['Volume'].clip(lower=1) # Avoid log(0) or division by zero

    if df.empty:
        print("Error: DataFrame is empty after cleaning.")
        return None

    print("Data loading and preparation complete.")
    return df

def engineer_features(df, window_size):
    """
    Engineers time-series features like log returns, volatility, momentum,
    and volume changes based on the specified window size.

    Args:
        df (pandas.DataFrame): The input DataFrame with price and volume data.
        window_size (int): The rolling window size for calculations.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame) or tuple(None, None):
            - df_features: DataFrame with engineered features.
            - df_aligned: Original DataFrame aligned to the feature index.
            Returns (None, None) if errors occur.
    """
    print(f"\n--- Engineering Features (Window Size: {window_size}) ---")
    if PRICE_COLUMN not in df.columns or 'Volume' not in df.columns:
        print("Error: Required columns for feature engineering (Price, Volume) are missing.")
        return None, None
    if df.empty:
        print("Error: Input DataFrame for feature engineering is empty.")
        return None, None

    df_eng = df.copy()
    try:
        # Log Returns
        df_eng['log_return'] = np.log(df_eng[PRICE_COLUMN] / df_eng[PRICE_COLUMN].shift(1))

        # Annualized Volatility (Standard deviation of log returns)
        df_eng['volatility'] = df_eng['log_return'].rolling(window=window_size, min_periods=window_size//2).std() * np.sqrt(252) # Added min_periods

        # Annualized Momentum (Mean of log returns)
        df_eng['momentum'] = df_eng['log_return'].rolling(window=window_size, min_periods=window_size//2).mean() * 252 # Added min_periods

        # Volume Change (Log ratio relative to rolling mean volume)
        rolling_mean_volume = df_eng['Volume'].rolling(window=window_size, min_periods=window_size//2).mean() # Added min_periods
        # Add small epsilon to avoid log(0) or division by zero
        df_eng['volume_log_ratio'] = np.log(df_eng['Volume'] / (rolling_mean_volume + 1e-9))

        # Replace potential infinities resulting from log/division issues
        df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)

    except Exception as e:
        print(f"Error during feature calculation: {e}")
        return None, None

    # Select features and handle NaNs introduced by rolling windows/shift
    features_list = ['log_return', 'volatility', 'momentum', 'volume_log_ratio']
    df_features = df_eng[features_list].copy()
    initial_rows = len(df_features)
    df_features.dropna(inplace=True)
    final_rows = len(df_features)
    print(f"Removed {initial_rows - final_rows} rows with NaNs after feature engineering.")

    if df_features.empty:
        print("Error: No data remaining after feature engineering and NaN removal.")
        return None, None

    # Align original data with the valid feature index
    df_aligned = df.loc[df_features.index].copy()

    print(f"Feature engineering complete. Shape: {df_features.shape}")
    return df_features, df_aligned

def scale_features(df_features):
    """
    Scales the features using StandardScaler.

    Args:
        df_features (pandas.DataFrame): DataFrame with features to scale.

    Returns:
        tuple(pandas.DataFrame, sklearn.preprocessing.StandardScaler) or tuple(None, None):
            - df_scaled: DataFrame with scaled features.
            - scaler: The fitted StandardScaler object.
            Returns (None, None) if scaling fails.
    """
    print("--- Scaling Features ---")
    if df_features is None or df_features.empty:
        print("Error: Cannot scale empty or None features DataFrame.")
        return None, None
    scaler = StandardScaler()
    try:
        scaled_features_array = scaler.fit_transform(df_features)
        df_scaled = pd.DataFrame(scaled_features_array, index=df_features.index, columns=df_features.columns)
        print("Features scaled successfully.")
        return df_scaled, scaler
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        return None, None

def plot_results(df_plot, y_col, label_col, title, filename):
    """
    Generates and saves a scatter plot colored by labels.

    Args:
        df_plot (pandas.DataFrame): Dataframe containing data to plot (incl. index, y_col, label_col).
        y_col (str): The column name for the y-axis (e.g., PRICE_COLUMN).
        label_col (str): The column name containing the category labels (e.g., 'regime', 'anomaly').
        title (str): The title for the plot.
        filename (str): The filename to save the plot (without extension).
    """
    if df_plot is None or df_plot.empty or label_col not in df_plot.columns:
        print(f"Warning: Skipping plot '{title}' due to missing data or label column '{label_col}'.")
        return

    plt.figure(figsize=(15, 7))
    unique_labels = sorted(df_plot[label_col].unique())
    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        subset = df_plot[df_plot[label_col] == label]
        if subset.empty:
            continue # Skip empty subsets

        plot_label = f'Label {label}'
        marker = 'o'
        size = 10
        plot_color = colors[i]
        alpha = 0.6
        zorder = 2 # Default z-order

        # Special handling for anomalies (-1)
        if label == -1 and 'anomaly' in label_col.lower():
             plot_label = f'Anomaly ({label})'
             plot_color = 'red'
             size = 30
             alpha = 0.7
             zorder = 5 # Draw anomalies on top

        plt.scatter(subset.index, subset[y_col], color=plot_color, label=plot_label,
                    alpha=alpha, s=size, marker=marker, zorder=zorder)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(f'{y_col} (Log Scale)')

    # Improve legend handling for many labels
    if len(unique_labels) > 10:
        plt.legend(title=label_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()

    plt.yscale('log') # Log scale is crucial for long-term stock data
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent legend overlap if outside

    # Ensure plots directory exists
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        save_path = os.path.join(PLOTS_DIR, f"{filename}.png")
        plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight'
        print(f"Plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot '{save_path}': {e}")
    plt.close() # Close the plot to free memory

def run_kmeans_regimes(df_scaled_features, df_aligned, k, feature_subset, window_size):
    """
    Runs K-Means clustering to identify market regimes, analyzes the regimes,
    and plots the results.

    Args:
        df_scaled_features (pandas.DataFrame): DataFrame with scaled features.
        df_aligned (pandas.DataFrame): Original DataFrame aligned with features.
        k (int): The number of clusters (regimes).
        feature_subset (list): List of feature names to use for clustering.
        window_size (int): The window size used for feature engineering (for labeling).

    Returns:
        pandas.DataFrame: df_aligned updated with the regime labels, or original if error.
    """
    if df_scaled_features is None or df_aligned is None:
        print(f"Error: Input data is None for K-Means (K={k}, Win={window_size}). Skipping.")
        return df_aligned if df_aligned is not None else pd.DataFrame()

    title_suffix = f"K={k}_Win={window_size}"
    print(f"\n--- Running K-Means Regime Detection ({title_suffix}) ---")
    regime_col_name = f'regime_{title_suffix}' # Unique column name per run

    # Ensure feature subset exists
    missing_features = [f for f in feature_subset if f not in df_scaled_features.columns]
    if missing_features:
        print(f"Error: Features {missing_features} not found in scaled data for K={k}. Skipping.")
        return df_aligned

    regime_features_data = df_scaled_features[feature_subset]

    try:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
        # Ensure data passed to fit is clean
        if regime_features_data.isnull().values.any():
             print(f"Warning: NaNs found in features for K-Means (K={k}). Attempting to fit anyway.")
        kmeans.fit(regime_features_data)
        # Add labels to the *aligned* DataFrame
        df_aligned_out = df_aligned.copy() # Avoid modifying original df passed in
        df_aligned_out[regime_col_name] = kmeans.labels_
        print(f"K-Means fitting complete. Regime labels added as '{regime_col_name}'.")

        # Analyze Regimes (Mean Scaled Feature Values)
        print("\nRegime Analysis (Mean Scaled Feature Values):")
        # Group by the newly added column in the output df
        regime_scaled_means = regime_features_data.groupby(df_aligned_out[regime_col_name]).mean()
        print(regime_scaled_means)

        # Plot Regimes
        plot_title = f'AAPL Price with Market Regimes ({title_suffix})'
        plot_filename = f'regimes_{title_suffix}'
        # Pass the DataFrame with the new labels for plotting
        plot_results(df_aligned_out, PRICE_COLUMN, regime_col_name, plot_title, plot_filename)

        return df_aligned_out # Return the updated DataFrame

    except Exception as e:
        print(f"Error during K-Means execution for K={k}, Win={window_size}: {e}")
        return df_aligned # Return original DataFrame on error

def run_anomaly_detection(df_scaled_features, df_aligned, df_features_unscaled, method, window_size):
    """
    Runs a specified anomaly detection method (IsolationForest, LOF, ZScore)
    and plots the results.

    Args:
        df_scaled_features (pandas.DataFrame): DataFrame with scaled features.
        df_aligned (pandas.DataFrame): Original DataFrame aligned with features.
        df_features_unscaled (pandas.DataFrame): Original (unscaled) features aligned with features.
        method (str): The anomaly detection method ('IsolationForest', 'LOF', 'ZScore').
        window_size (int): The window size used for feature engineering (for labeling).

    Returns:
        pandas.DataFrame: df_aligned updated with anomaly labels/scores, or original if error.
    """
    if df_scaled_features is None or df_aligned is None or df_features_unscaled is None:
        print(f"Error: Input data is None for Anomaly Detection ({method}, Win={window_size}). Skipping.")
        return df_aligned if df_aligned is not None else pd.DataFrame()

    title_suffix = f"{method}_Win={window_size}"
    print(f"\n--- Running Anomaly Detection ({title_suffix}) ---")
    features_for_anomaly = df_scaled_features.copy()
    anomaly_col_name = f'anomaly_{title_suffix}' # Unique column name
    score_col_name = f'score_{title_suffix}' # Unique column name

    # Create a copy to avoid modifying the original DataFrame passed in
    df_aligned_out = df_aligned.copy()

    try:
        if method == 'IsolationForest':
            model = IsolationForest(contamination='auto', random_state=RANDOM_STATE, n_estimators=100)
            model.fit(features_for_anomaly)
            # decision_function: lower score = more anomalous
            df_aligned_out[score_col_name] = model.decision_function(features_for_anomaly)
            df_aligned_out[anomaly_col_name] = model.predict(features_for_anomaly) # -1 for anomalies, 1 for inliers
        elif method == 'LOF':
            model = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination='auto', novelty=False)
            df_aligned_out[anomaly_col_name] = model.fit_predict(features_for_anomaly) # -1 for anomalies, 1 for inliers
            # Store negative_outlier_factor_ (higher value means less anomalous, so lower means more anomalous)
            df_aligned_out[score_col_name] = model.negative_outlier_factor_
        elif method == 'ZScore':
            if ZSCORE_FEATURE not in features_for_anomaly.columns:
                 print(f"Error: Z-Score feature '{ZSCORE_FEATURE}' not found for Win={window_size}. Skipping.")
                 return df_aligned # Return original df on error
            # Calculate Z-scores for the specified feature
            feature_data = features_for_anomaly[ZSCORE_FEATURE]
            z_scores = (feature_data - feature_data.mean()) / feature_data.std()
            # Assign anomaly label based on threshold
            df_aligned_out[anomaly_col_name] = np.where(np.abs(z_scores) > ZSCORE_THRESHOLD, -1, 1) # -1 for anomalies
            # Store negative absolute Z-score (lower = more anomalous)
            df_aligned_out[score_col_name] = -np.abs(z_scores)
        else:
            print(f"Error: Unknown anomaly detection method: {method}")
            return df_aligned # Return original df on error

        print(f"Anomaly detection ({method}) complete. Labels/Scores added.")

        # Analyze and Plot Anomalies
        anomalies = df_aligned_out[df_aligned_out[anomaly_col_name] == -1]
        print(f"Number of anomalies detected by {method} (Win={window_size}): {len(anomalies)}")
        if not anomalies.empty:
            print(f"Top 5 Anomalies detected by {method} (Win={window_size}, by date):")
            # Show relevant original features for context by joining with unscaled features
            display_cols = [PRICE_COLUMN, anomaly_col_name]
            if score_col_name in df_aligned_out.columns:
                 display_cols.append(score_col_name)
            # Join anomaly info with original unscaled features for better interpretation
            anomaly_details = df_features_unscaled.loc[anomalies.head().index].join(anomalies[display_cols])
            print(anomaly_details)

        plot_title = f'AAPL Price with Detected Anomalies ({title_suffix})'
        plot_filename = f'anomalies_{title_suffix}'
        # Pass the DataFrame with the new labels for plotting
        plot_results(df_aligned_out, PRICE_COLUMN, anomaly_col_name, plot_title, plot_filename)

        return df_aligned_out # Return the updated DataFrame

    except Exception as e:
        print(f"Error during {method} anomaly detection for Win={window_size}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return df_aligned # Return original DataFrame on error


# --- Main Execution Script ---
if __name__ == "__main__":
    print("Starting AAPL Market Pattern Analysis...")
    print(f"Saving plots to: {os.path.abspath(PLOTS_DIR)}") # Show absolute path

    # 1. Load and Prepare Data
    df_raw = load_and_prep_data(CSV_PATH)

    # Initialize DataFrames to store results from different window sizes
    df_analysis_orig = None
    df_analysis_new = None

    if df_raw is not None:
        # --- Analysis with ORIGINAL window size ---
        print(f"\n======= Starting Analysis with Window Size: {ORIGINAL_WINDOW_SIZE} =======")
        df_features_orig, df_aligned_orig = engineer_features(df_raw, ORIGINAL_WINDOW_SIZE)

        if df_features_orig is not None:
            df_scaled_orig, scaler_orig = scale_features(df_features_orig)

            if df_scaled_orig is not None and df_aligned_orig is not None:
                # Keep a copy of the aligned data for this window size
                df_analysis_orig = df_aligned_orig.copy()

                # Run K-Means for different K values
                all_ks = [DEFAULT_REGIME_K] + REGIME_KS_TO_TRY # e.g., [4, 3, 5, 6]
                for k in all_ks:
                    # Update df_analysis_orig with results from each k-means run
                    df_analysis_orig = run_kmeans_regimes(df_scaled_orig, df_analysis_orig, k, REGIME_FEATURES, ORIGINAL_WINDOW_SIZE)

                # Run different anomaly detection methods
                for method in ANOMALY_METHODS_TO_TRY:
                    # Update df_analysis_orig with results from each anomaly run
                    df_analysis_orig = run_anomaly_detection(df_scaled_orig, df_analysis_orig, df_features_orig, method, ORIGINAL_WINDOW_SIZE)
            else:
                 print("Skipping analysis for original window size due to scaling or alignment error.")
        else:
             print("Skipping analysis for original window size due to feature engineering error.")


        # --- Analysis with NEW window size ---
        print(f"\n======= Starting Analysis with Window Size: {NEW_WINDOW_SIZE} =======")
        df_features_new, df_aligned_new = engineer_features(df_raw, NEW_WINDOW_SIZE)

        if df_features_new is not None:
            df_scaled_new, scaler_new = scale_features(df_features_new)

            if df_scaled_new is not None and df_aligned_new is not None:
                 # Keep a copy of the aligned data for this window size
                df_analysis_new = df_aligned_new.copy()

                # Run K-Means with default K
                df_analysis_new = run_kmeans_regimes(df_scaled_new, df_analysis_new, DEFAULT_REGIME_K, REGIME_FEATURES, NEW_WINDOW_SIZE)

                # Run Isolation Forest (as an example) with new window features
                df_analysis_new = run_anomaly_detection(df_scaled_new, df_analysis_new, df_features_new, 'IsolationForest', NEW_WINDOW_SIZE)

                # You could run other anomaly detectors here too if desired for the new window size
                # for method in ANOMALY_METHODS_TO_TRY:
                #     df_analysis_new = run_anomaly_detection(df_scaled_new, df_analysis_new, df_features_new, method, NEW_WINDOW_SIZE)
            else:
                print("Skipping analysis for new window size due to scaling or alignment error.")
        else:
            print("Skipping analysis for new window size due to feature engineering error.")


        print("\n======= Analysis Complete =======")
        # Display final DataFrames with added labels/scores (optional sample)
        if df_analysis_orig is not None:
            print("\nFinal DataFrame Sample (Original Window - showing added columns):")
            # List columns added during analysis
            added_cols_orig = [col for col in df_analysis_orig.columns if 'regime_' in col or 'anomaly_' in col or 'score_' in col]
            print(df_analysis_orig[added_cols_orig].head())

        if df_analysis_new is not None:
            print("\nFinal DataFrame Sample (New Window - showing added columns):")
            added_cols_new = [col for col in df_analysis_new.columns if 'regime_' in col or 'anomaly_' in col or 'score_' in col]
            print(df_analysis_new[added_cols_new].head())

    else:
        print("\nExecution halted due to data loading errors.")

