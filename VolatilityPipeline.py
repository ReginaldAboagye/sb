# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:25:04 2025

@author: regay
"""

# -*- coding: utf-8 -*-
"""
Enhanced Volatility Forecasting Pipeline with Robust Data Handling
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pymc as pm
import shap
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import requests
from datetime import datetime, timedelta

class VolatilityPipeline:
    def __init__(self, ticker, start_date, end_date, auto_adjust=True):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.auto_adjust = auto_adjust
        self.models = {}
        self.explainers = {}
        
    def _robust_download(self, max_retries=5):
        """Robust data download with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=self.auto_adjust,
                    progress=False
                )
                if not data.empty:
                    return data
            except (ConnectionError, requests.exceptions.RequestException) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download data after {max_retries} attempts") from e
                print(f"Retry {attempt+1}/{max_retries} due to {e}")
                continue
        return pd.DataFrame()

    def _prepare_data(self, lags=20):
        """Enhanced data preparation with validation"""
        data = self._robust_download()
        
        if data.empty:
            raise ValueError(f"No data retrieved for {self.ticker}. Check ticker symbol and date range.")
            
        # Handle different price columns
        price_col = 'Close' if self.auto_adjust else 'Adj Close'
        if price_col not in data.columns:
            available_cols = ", ".join(data.columns)
            raise KeyError(f"Price column {price_col} not found. Available columns: {available_cols}")
            
        data['Returns'] = data[price_col].pct_change().dropna()
        
        # Feature engineering with validation
        if len(data) < lags + 22:  # 21 for volatility window + 1 shift
            raise ValueError(f"Insufficient data points. Need at least {lags+22} days, got {len(data)}")
            
        for lag in range(1, lags+1):
            data[f'Lag_{lag}'] = data['Returns'].shift(lag)
        
        data['Volatility'] = data['Returns'].rolling(21).std().shift(-1) * np.sqrt(252)
        data.dropna(inplace=True)
        
        if data.empty:
            raise ValueError("Data preprocessing resulted in empty dataset. Check calculation windows.")
            
        X = data[[f'Lag_{lag}' for lag in range(1, lags+1)] + ['Returns']]
        y = data['Volatility']
        
        return train_test_split(
            StandardScaler().fit_transform(X), y, 
            test_size=0.2, random_state=42
        )

    def train_svr(self):
        """Train SVR model with Bayesian hyperparameter tuning"""
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # Store test set for later use
        self.X_test, self.y_test = X_test, y_test
        
        # Grid search for optimal parameters
        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(
            estimator=svr, 
            param_grid={
                'C': [0.01, 0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5, 1, 10],
                'gamma': ['scale', 'auto']
            },
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['svr'] = grid_search.best_estimator_
        self.y_pred = self.models['svr'].predict(X_test)
        
        return self

    def bayesian_volatility(self):
        """Bayesian stochastic volatility model for residuals"""
        residuals = self.y_test - self.y_pred
        
        with pm.Model() as volatility_model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=1)
            phi = pm.Beta('phi', alpha=2, beta=2)
            sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)
            
            # Stochastic volatility process
            h = pm.AR(
                'h', 
                rho=phi, 
                sigma=sigma_eta, 
                init_dist=pm.Normal.dist(0, 1), 
                shape=len(residuals)
            )
            
            # Observation model
            pm.Normal(
                'residuals', 
                mu=mu, 
                sigma=pm.math.exp(h/2), 
                observed=residuals
            )
            
            # Sampling
            self.trace = pm.sample(
                2000, tune=1000, 
                chains=4, target_accept=0.95,
                random_seed=42
            )
            
        return self

    def kalman_smoothing(self):
        """Adaptive Kalman Filter for volatility smoothing"""
        kf = KalmanFilter(
            initial_state_mean=0,
            initial_state_covariance=1,
            transition_matrices=1,
            observation_matrices=1,
            transition_covariance=1e-5,
            observation_covariance=1e-5
        )
        
        filtered_means, _ = kf.filter(self.y_pred)
        self.smoothed_vol = filtered_means.squeeze()
        
        return self

    def explain_models(self):
        """Model explanation using SHAP values"""
        explainer = shap.KernelExplainer(
            self.models['svr'].predict, 
            self.X_test[:100]
        )
        self.shap_values = explainer.shap_values(self.X_test[:100])
        
        return self

    def risk_analysis(self):
        """Generate risk management insights"""
        # Convert posterior samples to numpy array
        h_samples = self.trace.posterior['h'].values
        h_flat = h_samples.reshape(-1, h_samples.shape[-1])
        
        # Value at Risk calculations
        var_95 = np.percentile(h_flat, 5, axis=0)
        cvar_95 = np.array([
            h_flat[:, t][h_flat[:, t] <= var_95[t]].mean()
            for t in range(h_flat.shape[1])
        ])
        
        # Safely format values
        current_vol = f"{self.y_pred[-1]:.2%}" if not np.isnan(self.y_pred[-1]) else "N/A"
        var_display = f"{var_95[-1]:.2%}" if not np.isnan(var_95[-1]) else "N/A"
        
        try:
            cvar_display = f"{cvar_95[-1]:.2%}"
        except (TypeError, ValueError):
            cvar_display = "N/A"
        
        # Feature importance analysis
        feature_importance = pd.Series(
            np.abs(self.shap_values).mean(0),
            index=[f'Lag_{i}' for i in range(1, 21)] + ['Returns']
        ).sort_values(ascending=False)
        
        # Generate report
        report = f"""
        RISK MANAGEMENT REPORT - {self.ticker}
        
        1. Volatility Forecast:
        - Current volatility: {current_vol}
        - 95% Confidence Interval: [{np.percentile(h_flat[:, -1], 2.5):.2%}, 
                                  {np.percentile(h_flat[:, -1], 97.5):.2%}]
        
        2. Risk Metrics:
        - 95% Value at Risk: {var_display}
        - Conditional VaR: {cvar_display}
        - Stress Scenario Impact: {self._stress_test():.2%}
        
        3. Key Drivers:
        {feature_importance.head(3).to_string()}
        
        4. Recommendations:
        {self._generate_recommendations()}
        """
        
        return report

    def _stress_test(self):
        """Stress testing for extreme market conditions"""
        crisis_data = self.X_test.mean(0) * 1.5  # Simulate extreme moves
        return self.models['svr'].predict([crisis_data])[0]

    def _generate_recommendations(self):
        """Generate investment recommendations"""
        recommendations = [
            "Maintain dynamic hedging strategy",
            "Consider volatility-linked derivatives",
            "Rebalance portfolio towards low-beta assets"
        ]
        return "\n".join(f"- {rec}" for rec in recommendations)

    def visualize(self):
        """Create explanatory visualizations"""
        plt.figure(figsize=(12, 6))
        
        # Get posterior quantiles
        h_samples = self.trace.posterior['h'].values
        h_flat = h_samples.reshape(-1, h_samples.shape[-1])
        
        plt.plot(self.y_test.values, label='Actual Volatility')
        plt.plot(self.y_pred, label='SVR Prediction', alpha=0.7)
        plt.plot(self.smoothed_vol, label='Kalman Filtered', linestyle='--')
        
        # Plot 95% CI using quantiles
        plt.fill_between(
            range(len(self.y_pred)),
            np.percentile(h_flat, 2.5, axis=0),
            np.percentile(h_flat, 97.5, axis=0),
            alpha=0.2, label='95% CI'
        )
        
        plt.title(f"{self.ticker} Volatility Forecast")
        plt.legend()
        plt.show()
        
        # SHAP plot
        shap.summary_plot(self.shap_values, self.X_test[:100], plot_type='dot')
        
        def risk_metrics(self):
            return {
                'Value_at_Risk_95': np.percentile(self.trace['h'], 5),
                'Expected_Shortfall': self.trace['h'][self.trace['h'] <= var_95].mean(),
                'Volatility_Persistence': self.trace['phi'].mean()
                }
    
        def save_pipeline(self, path):
            joblib.dump({
                'model': self.models,
                'scaler': self.scaler,
                'shap_values': self.shap_values
            }, path)
        
        def load_pipeline(self, path):
            artifacts = joblib.load(path)
            self.models = artifacts['model']
            self.scaler = artifacts['scaler']
            self.shap_values = artifacts['shap_values']
            
        def monitor_performance(self):
            mse = mean_squared_error(self.y_test, self.y_pred)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            calibration = np.mean(self.y_test < np.percentile(self.trace['h'], 95))
            return {'MSE': mse, 'MAE': mae, 'Calibration': calibration}
            
        def retrain_pipeline(self, new_data):
            """Online learning updates"""
            self.models['svr'].partial_fit(new_data)
            self.bayesian_volatility()  # Update posterior estimates
            self.kalman_smoothing()
            return self
        
        def generate_trade_signals(self):
            signals = []
            current_vol = self.y_pred[-1]
            if current_vol > np.percentile(self.trace['h'], 90):
                signals.append("Activate volatility targeting strategy")
            if self.trace['phi'].mean() > 0.8:
                signals.append("Implement momentum-based hedging")
            return signals
        
        def generate_regulatory_report(self):
            return {
                'FRTB_SA': self._calculate_frtb_risk(),
                'CCAR_Scenarios': self._ccar_stress_tests(),
                'Volatility_Projections': self.y_pred[-30:]
            }
        

# Updated usage with validation
if __name__ == "__main__":
    try:
        # Test with extended date range and fallback
        pipeline = VolatilityPipeline("^DJI", "1990-01-01", datetime.today().strftime('%Y-%m-%d'),auto_adjust=False)
        
        (pipeline.train_svr()
                .bayesian_volatility()
                .kalman_smoothing()
                .explain_models()
                .visualize())
        
        print(pipeline.risk_analysis())
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Verify internet connection")
        print("2. Check ticker symbol on Yahoo Finance")
        print("3. Try a smaller date range (e.g., 2020-01-01 to 2023-01-01)")
        print("4. Test with alternative ticker (e.g., 'SPY')")