import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Time Series
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Technical Analysis
try:
    from ta import add_all_ta_features
    from ta.utils import dropna
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.trend import MACD, EMAIndicator, SMAIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volume import VolumeSMAIndicator, OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    print("Warning: TA library not available. Technical indicators will be limited.")
    TA_AVAILABLE = False

# Advanced ML (optional)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === Configuration ===
CONFIG = {
    'DATA_FOLDER': r"D:\Users\TANISH\Downloads\New folder",  # Update this path
    'PLOT_DIR': None,  # Will be set automatically
    'PREDICTION_HORIZONS': [1, 3, 5],  # Days ahead to predict
    'FEATURE_SELECTION_K': 20,  # Number of top features to select
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'CV_FOLDS': 5,
    'MIN_TRADING_DAYS': 100,  # Minimum days needed for analysis
    'VOLATILITY_WINDOW': 20,
    'TREND_WINDOW': 50,
    'VOLUME_WINDOW': 20
}

CONFIG['PLOT_DIR'] = os.path.join(CONFIG['DATA_FOLDER'], "advanced_plots")
os.makedirs(CONFIG['PLOT_DIR'], exist_ok=True)

class NiftyTradingSystem:
    """
    Complete NIFTY 50 Trading System with Advanced Analytics
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.data = None
        self.features = None
        self.targets = {}
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
    def detect_and_rename_columns(self, df):
        """Enhanced column detection with better mapping."""
        column_mappings = {
            'date': ['Date', 'DATE', 'date', 'Date ', 'DATE ', 'date ', 'Date Time', 'DateTime', 'timestamp'],
            'open': ['Open', 'OPEN', 'open', 'Open ', 'OPEN ', 'open ', 'Open Price', 'Opening Price'],
            'high': ['High', 'HIGH', 'high', 'High ', 'HIGH ', 'high ', 'High Price', 'Highest Price'],
            'low': ['Low', 'LOW', 'low', 'Low ', 'LOW ', 'low ', 'Low Price', 'Lowest Price'],
            'close': ['Close', 'CLOSE', 'close', 'Close ', 'CLOSE ', 'close ', 'Close Price', 'Closing Price', 'Last Price'],
            'volume': ['Volume', 'VOLUME', 'volume', 'Volume ', 'VOLUME ', 'volume ', 'Shares Traded', 'Shares Traded ', 'Total Traded Quantity', 'Qty'],
            'turnover': ['Turnover', 'TURNOVER', 'turnover', 'Turnover ', 'Turnover (₹ Cr)', 'Value', 'Total Value']
        }
        
        detected_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                # Exact match or contains match (after stripping)
                col_clean = col.strip()
                if col_clean in possible_names or any(pname.lower() in col_clean.lower() for pname in possible_names):
                    detected_columns[standard_name] = col
                    break
        
        print("📋 Detected columns:")
        for std_name, actual_name in detected_columns.items():
            print(f"  {std_name}: '{actual_name}'")
        
        # Rename columns
        rename_dict = {v: k.capitalize() for k, v in detected_columns.items()}
        df = df.rename(columns=rename_dict)
        
        return df, detected_columns

    def load_and_prepare_data(self):
        """Load and prepare all NIFTY data files."""
        print("📥 Loading NIFTY data...")
        
        # Find CSV files
        csv_files = glob.glob(os.path.join(self.config['DATA_FOLDER'], "*NIFTY*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(self.config['DATA_FOLDER'], "*.csv"))
            if csv_files:
                print(f"No NIFTY files found, using all CSV files: {len(csv_files)} files")
            else:
                raise FileNotFoundError(f"No CSV files found in {self.config['DATA_FOLDER']}")
        
        dfs = []
        for file in csv_files:
            try:
                print(f"📄 Processing: {os.path.basename(file)}")
                df = pd.read_csv(file)
                
                # Detect and rename columns
                df, detected_cols = self.detect_and_rename_columns(df)
                
                # Handle date column
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['SourceFile'] = os.path.basename(file)
                
                # Clean numeric columns
                for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('Cr', ''),
                            errors='coerce'
                        )
                
                dfs.append(df)
                print(f"✅ Successfully processed {os.path.basename(file)}")
                
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No valid data files could be processed")
        
        # Combine all data
        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.sort_values("Date").reset_index(drop=True)
        self.data = self.data.drop_duplicates(subset=['Date'], keep='last')
        
        print(f"\n🎯 Data Summary:")
        print(f"  📅 Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"  📊 Total Records: {len(self.data)}")
        print(f"  📈 Available Columns: {list(self.data.columns)}")
        
        return self.data

    def add_advanced_features(self):
        """Add comprehensive technical indicators and features."""
        print("\n🔧 Adding advanced technical features...")
        
        df = self.data.copy()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Volatility features
        for window in [5, 10, 20, 50]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window).std()
            df[f'MA_{window}'] = df['Close'].rolling(window).mean()
            df[f'MA_ratio_{window}'] = df['Close'] / df[f'MA_{window}']
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window).mean() if 'Volume' in df.columns else np.nan
            df[f'Turnover_MA_{window}'] = df['Turnover'].rolling(window).mean() if 'Turnover' in df.columns else np.nan
        
        # Price momentum features
        for period in [1, 3, 5, 10, 20]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
            df[f'Max_High_{period}d'] = df['High'].rolling(period).max()
            df[f'Min_Low_{period}d'] = df['Low'].rolling(period).min()
        
        # Advanced technical indicators
        if TA_AVAILABLE:
            try:
                # Trend indicators
                df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
                df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
                df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
                
                # MACD
                macd = MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()
                
                # RSI
                df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
                
                # Bollinger Bands
                bb = BollingerBands(df['Close'], window=20, window_dev=2)
                df['BB_Upper'] = bb.bollinger_hband()
                df['BB_Lower'] = bb.bollinger_lband()
                df['BB_Middle'] = bb.bollinger_mavg()
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                
                # Average True Range
                df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
                
                # Volume indicators (if volume available)
                if 'Volume' in df.columns:
                    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
                    df['Volume_SMA'] = VolumeSMAIndicator(df['Close'], df['Volume'], window=20).volume_sma()
                    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                
                # Stochastic Oscillator
                stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
                
            except Exception as e:
                print(f"⚠️ Warning: Some technical indicators failed: {e}")
        
        # Market regime features
        df['Trend_Regime'] = np.where(df['Close'] > df['MA_50'], 1, 0)  # Above/below long-term MA
        df['Volatility_Regime'] = pd.qcut(df['Volatility_20d'].fillna(0), q=3, labels=[0, 1, 2])  # Low/Med/High vol
        
        # Seasonality features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Lag features to avoid lookahead bias
        for lag in [1, 2, 3, 5]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else np.nan
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag) if 'RSI' in df.columns else np.nan
        
        # Market structure features
        df['New_High_20d'] = (df['Close'] == df['Close'].rolling(20).max()).astype(int)
        df['New_Low_20d'] = (df['Close'] == df['Close'].rolling(20).min()).astype(int)
        df['Days_Since_High'] = df.groupby((df['Close'] == df['Close'].rolling(20).max()).cumsum()).cumcount()
        df['Days_Since_Low'] = df.groupby((df['Close'] == df['Close'].rolling(20).min()).cumsum()).cumcount()
        
        self.data = df.reset_index()
        print(f"✅ Added {len(df.columns)} total features")
        return self.data

    def create_targets(self):
        """Create multiple prediction targets."""
        print("\n🎯 Creating prediction targets...")
        
        df = self.data.copy()
        self.targets = {}
        
        for horizon in self.config['PREDICTION_HORIZONS']:
            # Binary direction prediction
            future_return = df['Close'].pct_change(horizon).shift(-horizon)
            self.targets[f'Direction_{horizon}d'] = (future_return > 0).astype(int)
            
            # Significant movement prediction (>1% or >2%)
            self.targets[f'Significant_Up_{horizon}d'] = (future_return > 0.01).astype(int)
            self.targets[f'Large_Up_{horizon}d'] = (future_return > 0.02).astype(int)
            self.targets[f'Significant_Down_{horizon}d'] = (future_return < -0.01).astype(int)
            self.targets[f'Large_Down_{horizon}d'] = (future_return < -0.02).astype(int)
            
            # Volatility prediction
            future_vol = df['Returns'].rolling(horizon).std().shift(-horizon)
            current_vol = df['Returns'].rolling(20).std()
            self.targets[f'High_Volatility_{horizon}d'] = (future_vol > current_vol * 1.5).astype(int)
        
        # Add targets to dataframe
        for name, target in self.targets.items():
            df[name] = target
        
        self.data = df
        print(f"✅ Created {len(self.targets)} different prediction targets")
        return self.targets

    def prepare_features_and_targets(self):
        """Prepare clean feature matrix and target vectors."""
        print("\n🧹 Preparing features and targets...")
        
        # Select numeric features only (exclude text and date columns)
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Exclude target columns and lag-0 features that might cause leakage
        target_cols = [col for col in numeric_columns if any(target_name in col for target_name in ['Direction_', 'Significant_', 'Large_', 'High_Volatility_'])]
        exclude_cols = ['Close', 'Open', 'High', 'Low'] + target_cols  # Exclude raw OHLC to prevent leakage
        
        feature_cols = [col for col in numeric_columns if col not in exclude_cols]
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        valid_features = []
        for col in feature_cols:
            if self.data[col].isnull().sum() / len(self.data) < missing_threshold:
                valid_features.append(col)
        
        print(f"📊 Feature selection: {len(valid_features)} features out of {len(feature_cols)} candidates")
        
        # Create feature matrix
        X = self.data[valid_features].copy()
        X = X.fillna(X.median())  # Fill remaining NaN values
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.features = X
        self.feature_names = valid_features
        
        print(f"✅ Final feature matrix shape: {X.shape}")
        return X, self.targets

    def train_models(self, target_name='Direction_1d'):
        """Train multiple models with advanced techniques."""
        print(f"\n🤖 Training models for target: {target_name}")
        
        if target_name not in self.targets:
            raise ValueError(f"Target {target_name} not found")
        
        # Prepare data
        X = self.features.copy()
        y = self.targets[target_name].copy()
        
        # Remove rows with NaN targets
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < self.config['MIN_TRADING_DAYS']:
            raise ValueError(f"Not enough data points: {len(X)} < {self.config['MIN_TRADING_DAYS']}")
        
        # Time-based split (important for financial data)
        split_idx = int(len(X) * (1 - self.config['TEST_SIZE']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Data split: Train={len(X_train)}, Test={len(X_test)}")
        print(f"📈 Target distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"📈 Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        # Feature scaling
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with optimized parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.config['RANDOM_STATE'],
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.config['RANDOM_STATE']
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.config['RANDOM_STATE'],
                class_weight='balanced',
                max_iter=1000,
                C=1.0
            )
        }
        
        # Train and evaluate each model
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Create pipeline with feature selection
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=min(self.config['FEATURE_SELECTION_K'], X_train.shape[1]))),
                ('classification', model)
            ])
            
            # Handle class imbalance with SMOTE if available
            if IMBALANCED_AVAILABLE and name != 'Logistic Regression':  # LR already has class_weight
                smote_pipeline = ImbPipeline([
                    ('feature_selection', SelectKBest(f_classif, k=min(self.config['FEATURE_SELECTION_K'], X_train.shape[1]))),
                    ('sampling', SMOTE(random_state=self.config['RANDOM_STATE'], k_neighbors=3)),
                    ('classification', model)
                ])
                pipeline = smote_pipeline
            
            # Train model
            try:
                pipeline.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test_scaled)
                y_proba = pipeline.predict_proba(X_test_scaled)[:, 1] if hasattr(pipeline, 'predict_proba') else None
                
                # Calculate metrics
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'y_test': y_test
                }
                
                trained_models[name] = pipeline
                print(f"✅ {name}: Accuracy={results[name]['accuracy']:.3f}, F1={results[name]['f1']:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        # Create ensemble model
        if len(trained_models) >= 2:
            try:
                # Get base models for ensemble (without pipeline wrapper for voting)
                base_models = []
                for name, pipeline in trained_models.items():
                    if hasattr(pipeline, 'named_steps'):
                        base_models.append((name.lower().replace(' ', '_'), pipeline))
                
                if base_models:
                    ensemble = VotingClassifier(estimators=base_models, voting='soft')
                    ensemble.fit(X_train_scaled, y_train)
                    
                    y_pred_ensemble = ensemble.predict(X_test_scaled)
                    y_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
                    
                    results['Ensemble'] = {
                        'accuracy': accuracy_score(y_test, y_pred_ensemble),
                        'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
                        'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
                        'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
                        'auc': roc_auc_score(y_test, y_proba_ensemble),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble),
                        'y_pred': y_pred_ensemble,
                        'y_proba': y_proba_ensemble,
                        'y_test': y_test
                    }
                    
                    trained_models['Ensemble'] = ensemble
                    print(f"✅ Ensemble: Accuracy={results['Ensemble']['accuracy']:.3f}, F1={results['Ensemble']['f1']:.3f}")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not create ensemble model: {e}")
        
        # Store results
        self.models[target_name] = trained_models
        self.scalers[target_name] = scaler
        self.results[target_name] = results
        
        return results, trained_models

    def create_comprehensive_visualizations(self, target_name='Direction_1d'):
        """Create comprehensive visualizations and analysis."""
        print(f"\n📊 Creating visualizations for {target_name}...")
        
        if target_name not in self.results:
            print("❌ No results found. Train models first.")
            return
        
        results = self.results[target_name]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Price and Technical Analysis Chart
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        
        # Price and moving averages
        axes[0].plot(self.data['Date'], self.data['Close'], label='Close Price', linewidth=2)
        for ma in [20, 50]:
            if f'MA_{ma}' in self.data.columns:
                axes[0].plot(self.data['Date'], self.data[f'MA_{ma}'], label=f'MA-{ma}', alpha=0.7)
        axes[0].set_title('NIFTY 50 Price Action with Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        if 'Volume' in self.data.columns:
            axes[1].bar(self.data['Date'], self.data['Volume'], alpha=0.6, color='orange')
            axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in self.data.columns:
            axes[2].plot(self.data['Date'], self.data['RSI'], color='purple', linewidth=2)
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[2].set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('RSI')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in self.data.columns:
            axes[3].plot(self.data['Date'], self.data['MACD'], label='MACD', linewidth=2)
            if 'MACD_Signal' in self.data.columns:
                axes[3].plot(self.data['Date'], self.data['MACD_Signal'], label='Signal', linewidth=2)
            if 'MACD_Histogram' in self.data.columns:
                axes[3].bar(self.data['Date'], self.data['MACD_Histogram'], alpha=0.3, label='Histogram')
            axes[3].set_title('MACD (Moving Average Convergence Divergence)', fontsize=14, fontweight='bold')
            axes[3].set_ylabel('MACD')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'technical_analysis_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(results.keys())
        
        for i, metric in enumerate(metrics):
            values = [results[name][metric] for name in model_names]
            bars = axes[i].bar(model_names, values, color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
            axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrices for best model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        cm = results[best_model]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[5])
        axes[5].set_title(f'Confusion Matrix - {best_model}', fontweight='bold')
        axes[5].set_xlabel('Predicted')
        axes[5].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'model_performance_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance Analysis
        if 'Random Forest' in self.models[target_name]:
            try:
                model = self.models[target_name]['Random Forest']
                
                # Get feature importances
                if hasattr(model, 'named_steps') and hasattr(model.named_steps['classification'], 'feature_importances_'):
                    importances = model.named_steps['classification'].feature_importances_
                    selected_features = model.named_steps['feature_selection'].get_support(indices=True)
                    feature_names = [self.feature_names[i] for i in selected_features]
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(20)
                    
                    plt.figure(figsize=(12, 8))
                    sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
                    plt.title(f'Top 20 Feature Importances - Random Forest ({target_name})', fontsize=16, fontweight='bold')
                    plt.xlabel('Importance Score')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.config['PLOT_DIR'], f'feature_importance_{target_name}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"⚠️ Could not create feature importance plot: {e}")
        
        # 4. Prediction Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prediction probability distribution
        best_model_results = results[best_model]
        if best_model_results['y_proba'] is not None:
            axes[0, 0].hist(best_model_results['y_proba'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title(f'Prediction Probability Distribution - {best_model}', fontweight='bold')
            axes[0, 0].set_xlabel('Predicted Probability')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Prediction vs Actual over time (sample)
        sample_size = min(200, len(best_model_results['y_test']))
        sample_indices = np.random.choice(len(best_model_results['y_test']), sample_size, replace=False)
        
        axes[0, 1].plot(sample_indices, best_model_results['y_test'].iloc[sample_indices], 'o-', label='Actual', alpha=0.7)
        axes[0, 1].plot(sample_indices, best_model_results['y_pred'][sample_indices], 's-', label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual (Sample)', fontweight='bold')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Target Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model comparison radar chart
        metrics_for_radar = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False)
        
        ax_radar = plt.subplot(223, projection='polar')
        
        for model_name in model_names[:3]:  # Top 3 models
            values = [results[model_name][metric] for metric in metrics_for_radar]
            values += values[:1]  # Complete the circle
            angles_plot = np.concatenate([angles, [angles[0]]])
            ax_radar.plot(angles_plot, values, 'o-', linewidth=2, label=model_name)
            ax_radar.fill(angles_plot, values, alpha=0.1)
        
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(metrics_for_radar)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Model Performance Radar Chart', fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Learning curve simulation
        if len(self.data) > 500:
            train_sizes = np.linspace(100, len(self.data) * 0.8, 10).astype(int)
            train_scores = []
            
            X_full = self.features.copy()
            y_full = self.targets[target_name].copy()
            
            # Remove NaN values
            valid_indices = ~y_full.isnull()
            X_full = X_full[valid_indices]
            y_full = y_full[valid_indices]
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_full)
            
            quick_model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            for size in train_sizes:
                if size < len(X_scaled):
                    X_train_curve = X_scaled[:size]
                    y_train_curve = y_full[:size]
                    X_test_curve = X_scaled[size:min(size+100, len(X_scaled))]
                    y_test_curve = y_full[size:min(size+100, len(y_full))]
                    
                    if len(X_test_curve) > 10:
                        quick_model.fit(X_train_curve, y_train_curve)
                        score = quick_model.score(X_test_curve, y_test_curve)
                        train_scores.append(score)
                    else:
                        train_scores.append(0)
            
            axes[1, 1].plot(train_sizes[:len(train_scores)], train_scores, 'o-', linewidth=2, color='green')
            axes[1, 1].set_title('Learning Curve (Simulated)', fontweight='bold')
            axes[1, 1].set_xlabel('Training Set Size')
            axes[1, 1].set_ylabel('Test Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'prediction_analysis_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {self.config['PLOT_DIR']}")

    def backtest_strategy(self, target_name='Direction_1d', initial_capital=100000):
        """Perform backtesting with realistic trading simulation."""
        print(f"\n📈 Backtesting strategy for {target_name}...")
        
        if target_name not in self.results:
            print("❌ No results found. Train models first.")
            return None
        
        # Get best model results
        results = self.results[target_name]
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        best_results = results[best_model]
        
        print(f"🏆 Using best model: {best_model}")
        
        # Create backtest dataframe
        test_start_idx = int(len(self.data) * (1 - self.config['TEST_SIZE']))
        test_data = self.data.iloc[test_start_idx:].copy().reset_index(drop=True)
        
        # Align predictions with test data
        predictions = best_results['y_pred']
        probabilities = best_results['y_proba'] if best_results['y_proba'] is not None else np.ones(len(predictions)) * 0.6
        
        # Ensure same length
        min_len = min(len(test_data), len(predictions))
        test_data = test_data.iloc[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        
        # Trading simulation
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        portfolio_values = [capital]
        transaction_cost = 0.001  # 0.1% transaction cost
        
        for i in range(len(test_data) - 1):
            current_price = test_data.iloc[i]['Close']
            next_price = test_data.iloc[i + 1]['Close']
            prediction = predictions[i]
            confidence = probabilities[i]
            
            # Trading logic
            if prediction == 1 and confidence > 0.6 and position != 1:  # Buy signal
                if position == -1:  # Close short position
                    profit = (current_price - entry_price) / entry_price * capital * (-1)
                    capital += profit - (capital * transaction_cost)
                    trades.append({
                        'date': test_data.iloc[i]['Date'],
                        'action': 'cover_short',
                        'price': current_price,
                        'profit': profit,
                        'capital': capital
                    })
                
                # Open long position
                entry_price = current_price
                position = 1
                trades.append({
                    'date': test_data.iloc[i]['Date'],
                    'action': 'buy',
                    'price': current_price,
                    'profit': 0,
                    'capital': capital
                })
                
            elif prediction == 0 and confidence > 0.6 and position != -1:  # Sell signal
                if position == 1:  # Close long position
                    profit = (current_price - entry_price) / entry_price * capital
                    capital += profit - (capital * transaction_cost)
                    trades.append({
                        'date': test_data.iloc[i]['Date'],
                        'action': 'sell',
                        'price': current_price,
                        'profit': profit,
                        'capital': capital
                    })
                
                # Open short position
                entry_price = current_price
                position = -1
                trades.append({
                    'date': test_data.iloc[i]['Date'],
                    'action': 'short',
                    'price': current_price,
                    'profit': 0,
                    'capital': capital
                })
            
            # Calculate portfolio value
            if position == 1:  # Long position
                portfolio_value = capital + (next_price - entry_price) / entry_price * capital
            elif position == -1:  # Short position
                portfolio_value = capital + (entry_price - next_price) / entry_price * capital
            else:  # No position
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = test_data['Close'].pct_change().dropna()
        
        # Ensure same length
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        benchmark_return = (test_data.iloc[-1]['Close'] - test_data.iloc[0]['Close']) / test_data.iloc[0]['Close']
        
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
        max_drawdown = (np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)
        max_dd = np.max(max_drawdown) if len(max_drawdown) > 0 else 0
        
        win_rate = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
        
        backtest_results = {
            'initial_capital': initial_capital,
            'final_capital': portfolio_values[-1],
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'portfolio_values': portfolio_values,
            'trades': trades
        }
        
        # Create backtest visualization
        plt.figure(figsize=(15, 10))
        
        # Portfolio performance
        plt.subplot(2, 2, 1)
        dates = test_data['Date'][:len(portfolio_values)]
        plt.plot(dates, portfolio_values, label='Strategy', linewidth=2, color='blue')
        benchmark_values = initial_capital * (1 + test_data['Close'].pct_change().fillna(0).cumsum())[:len(dates)]
        plt.plot(dates, benchmark_values, label='Buy & Hold', linewidth=2, color='red', alpha=0.7)
        plt.title('Portfolio Performance Comparison', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Drawdown
        plt.subplot(2, 2, 2)
        plt.fill_between(dates, max_drawdown[:len(dates)] * 100, 0, alpha=0.3, color='red')
        plt.plot(dates, max_drawdown[:len(dates)] * 100, color='red', linewidth=2)
        plt.title('Drawdown Analysis', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        if len(portfolio_returns) > 30:
            try:
                returns_df = pd.DataFrame({
                    'returns': portfolio_returns,
                    'date': dates[1:len(portfolio_returns)+1]
                })
                returns_df['year'] = returns_df['date'].dt.year
                returns_df['month'] = returns_df['date'].dt.month
                
                monthly_returns = returns_df.groupby(['year', 'month'])['returns'].sum().reset_index()
                monthly_pivot = monthly_returns.pivot(index='year', columns='month', values='returns') * 100
                
                plt.subplot(2, 2, 3)
                sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Returns (%)'})
                plt.title('Monthly Returns Heatmap', fontweight='bold')
            except:
                plt.subplot(2, 2, 3)
                plt.text(0.5, 0.5, 'Monthly returns\nanalysis unavailable', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Performance metrics summary
        plt.subplot(2, 2, 4)
        metrics_text = f"""
        PERFORMANCE SUMMARY
        
        Total Return: {total_return:.2%}
        Benchmark Return: {benchmark_return:.2%}
        Excess Return: {total_return - benchmark_return:.2%}
        
        Sharpe Ratio: {sharpe_ratio:.2f}
        Max Drawdown: {max_dd:.2%}
        Win Rate: {win_rate:.2%}
        
        Total Trades: {len(trades)}
        Final Capital: ₹{portfolio_values[-1]:,.2f}
        """
        
        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'backtest_results_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Backtest Results Summary:")
        print(f"  💰 Initial Capital: ₹{initial_capital:,}")
        print(f"  💰 Final Capital: ₹{portfolio_values[-1]:,.2f}")
        print(f"  📈 Total Return: {total_return:.2%}")
        print(f"  📊 Benchmark Return: {benchmark_return:.2%}")
        print(f"  🎯 Excess Return: {total_return - benchmark_return:.2%}")
        print(f"  📉 Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  📉 Max Drawdown: {max_dd:.2%}")
        print(f"  🎯 Win Rate: {win_rate:.2%}")
        print(f"  📊 Total Trades: {len(trades)}")
        
        return backtest_results

    def save_model(self, target_name='Direction_1d', filename=None):
        """Save trained models and scalers."""
        if filename is None:
            filename = f"nifty_model_{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = os.path.join(self.config['PLOT_DIR'], filename)
        
        model_data = {
            'models': self.models.get(target_name, {}),
            'scaler': self.scalers.get(target_name, None),
            'feature_names': self.feature_names,
            'target_name': target_name,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
        return filepath

    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n📝 Generating comprehensive report...")
        
        report_path = os.path.join(self.config['PLOT_DIR'], 'NIFTY_Analysis_Report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NIFTY 50 ADVANCED TRADING SYSTEM ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}\n")
            f.write(f"Total Records: {len(self.data):,}\n")
            f.write(f"Total Features: {len(self.feature_names)}\n")
            f.write(f"Target Variables: {len(self.targets)}\n\n")
            
            # Results for each target
            for target_name, results in self.results.items():
                f.write(f"RESULTS FOR {target_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                best_model = max(results.keys(), key=lambda x: results[x]['f1'])
                f.write(f"Best Model: {best_model}\n")
                
                for model_name, metrics in results.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                    f.write(f"  Precision: {metrics['precision']:.3f}\n")
                    f.write(f"  Recall: {metrics['recall']:.3f}\n")
                    f.write(f"  F1 Score: {metrics['f1']:.3f}\n")
                    f.write(f"  AUC Score: {metrics['auc']:.3f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
        
        print(f"📄 Report saved to: {report_path}")
        return report_path

def main():
    """Main execution function."""
    print("🚀 NIFTY 50 Advanced Trading System")
    print("=" * 50)
    
    try:
        # Initialize system
        system = NiftyTradingSystem()
        
        # Load and prepare data
        system.load_and_prepare_data()
        
        # Add features
        system.add_advanced_features()
        
        # Create targets
        system.create_targets()
        
        # Prepare features and targets
        system.prepare_features_and_targets()
        
        # Train models for different prediction horizons
        for target in ['Direction_1d', 'Direction_3d', 'Significant_Up_1d']:
            if target in system.targets:
                print(f"\n{'='*60}")
                print(f"TRAINING MODELS FOR: {target}")
                print(f"{'='*60}")
                
                # Train models
                system.train_models(target)
                
                # Create visualizations
                system.create_comprehensive_visualizations(target)
                
                # Perform backtesting
                system.backtest_strategy(target)
                
                # Save model
                system.save_model(target)
        
        # Generate final report
        system.generate_report()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 All results saved to: {system.config['PLOT_DIR']}")
        print(f"📊 Check the plots and report for detailed insights.")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
