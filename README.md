# Nifty50-trading-system
A machine learning system for predicting NIFTY 50 index movements using technical analysis and ensemble models. A machine learning-based trading system for the NIFTY 50 index, featuring technical indicators, ensemble models, and back testing. Built with Python, scikit-learn, and TA-lib.
## Features
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more.
- **Multiple Targets**: Predicts 1-day, 3-day, and 5-day price movements.
- **Modeling**: Random Forest, Gradient Boosting, Logistic Regression, and Ensemble.
- **Backtesting**: Realistic trading simulation with transaction costs.
- **Visualization**: Comprehensive plots for performance analysis.
# Core Data & Computation
numpy>=1.20.0
pandas>=1.2.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
joblib>=1.0.0

# Technical Analysis
ta>=0.7.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0  # Optional for interactive plots

# Time Series Analysis
statsmodels>=0.12.0

# Utilities
tqdm>=4.0.0  # Progress bars
python-dotenv>=0.19.0  # For environment variables (optional)

# Optional (Uncomment if needed)
# pyportfolioopt>=1.4.1  # Advanced portfolio optimization
# backtrader>=1.9.76.123  # Alternative backtesting engine
