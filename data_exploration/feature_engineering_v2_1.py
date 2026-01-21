"""
Production-Grade Feature Engineering Pipeline (v2)
==================================================

Clean, simple, production-ready feature engineering for churn prediction.
Uses class_weight for imbalance handling (no SMOTE).

Usage:
    from feature_engineering import ChurnFeatureEngineer
    
    # Initialize and fit
    fe = ChurnFeatureEngineer()
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    # Train model with class_weight (handles imbalance)
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train_transformed, y_train)
    
    # Save for production
    fe.save('feature_engineer.joblib')
"""

import pandas as pd
import numpy as np
import joblib
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    FunctionTransformer,
    KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline for bank churn prediction.

    Features created:
    - Scaled numerical features (StandardScaler, RobustScaler)
    - Binned Age (KBinsDiscretizer) 
    - Log-transformed Balance
    - One-hot encoded categoricals
    - Binary flags (HasBalance, IsSenior, etc.)
    - Interaction features (Germany_Female, Senior_Inactive, etc.)
    - Composite RiskScore

    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    feature_names_ : list
        Names of output features after transformation.

    preprocessor_ : ColumnTransformer
        Fitted sklearn ColumnTransformer.

    custom_scaler_ : StandardScaler
        Scaler for continuous custom features.

    Example
    -------
    >>> fe = ChurnFeatureEngineer()
    >>> X_train_fe = fe.fit_transform(X_train)
    >>> X_test_fe = fe.transform(X_test)
    >>> 
    >>> # Use class_weight for imbalance (NOT SMOTE)
    >>> model = RandomForestClassifier(class_weight='balanced')
    >>> model.fit(X_train_fe, y_train)
    """

    # Column configurations
    DROP_COLUMNS = ['Tenure', 'EstimatedSalary', 'HasCrCard']
    NUMERIC_STANDARD = ['CreditScore']
    NUMERIC_ROBUST = ['Age']
    NUMERIC_LOG = ['Balance']
    CATEGORICAL = ['Geography', 'Gender']
    CATEGORICAL_SPECIAL = ['NumOfProducts']
    BINARY = ['IsActiveMember']
    CONTINUOUS_CUSTOM = ['Age_Log', 'BalancePerProduct', 'RiskScore']

    # Risk score weights (from EDA lift analysis)
    RISK_WEIGHTS = {
        'IsGermany': 1.6,
        'IsFemale': 1.2,
        'IsInactive': 1.3,
        'IsSenior': 1.5,
        'Has3PlusProducts': 4.2,
        'Senior_Inactive': 2.0
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor_ = None
        self.custom_scaler_ = None
        self.feature_names_ = None
        self._is_fitted = False

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the sklearn ColumnTransformer."""
        return ColumnTransformer(
            transformers=[
                # Standard scaling for CreditScore
                ('num_standard', StandardScaler(), self.NUMERIC_STANDARD),

                # Robust scaling for Age (handles outliers)
                ('num_robust', RobustScaler(), self.NUMERIC_ROBUST),

                # Binning for Age (captures non-linear relationship)
                ('age_bins', KBinsDiscretizer(
                    n_bins=5,
                    encode='onehot-dense',
                    strategy='quantile'
                ), self.NUMERIC_ROBUST),

                # Log transform + scale for Balance
                ('num_log', Pipeline([
                    ('log1p', FunctionTransformer(
                        np.log1p,
                        validate=True,
                        feature_names_out='one-to-one'
                    )),
                    ('scale', StandardScaler())
                ]), self.NUMERIC_LOG),

                # One-hot encoding for Geography, Gender
                ('cat', OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore'
                ), self.CATEGORICAL),

                # One-hot for NumOfProducts (non-linear relationship)
                ('products', OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                ), self.CATEGORICAL_SPECIAL),

                # Pass through binary columns
                ('binary', 'passthrough', self.BINARY),

                # Drop low-signal columns
                ('drop', 'drop', self.DROP_COLUMNS)
            ],
            remainder='drop'
        )

    def _create_custom_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create custom features from original data.

        Parameters
        ----------
        X : DataFrame
            Original feature DataFrame.

        Returns
        -------
        DataFrame
            Custom features.
        """
        features = pd.DataFrame(index=range(len(X)))

        # Binary Flags
        features['HasBalance'] = (X['Balance'].values > 0).astype(int)
        features['IsSenior'] = (X['Age'].values > 50).astype(int)
        features['Has3PlusProducts'] = (X['NumOfProducts'].values >= 3).astype(int)
        features['IsInactive'] = (X['IsActiveMember'].values == 0).astype(int)
        features['IsGermany'] = (X['Geography'].values == 'Germany').astype(int)
        features['IsFemale'] = (X['Gender'].values == 'Female').astype(int)

        # Interactions (high-risk segments from EDA)
        features['Germany_Female'] = features['IsGermany'] * features['IsFemale']
        features['Senior_Inactive'] = features['IsSenior'] * features['IsInactive']
        features['Germany_Senior'] = features['IsGermany'] * features['IsSenior']

        # Log transform for Age (better correlation than Age²)
        features['Age_Log'] = np.log1p(X['Age'].values)

        # Ratio feature
        num_products = np.maximum(X['NumOfProducts'].values, 1)
        features['BalancePerProduct'] = X['Balance'].values / num_products

        # Composite Risk Score (weights from lift analysis)
        features['RiskScore'] = (
            features['IsGermany'] * self.RISK_WEIGHTS['IsGermany'] +
            features['IsFemale'] * self.RISK_WEIGHTS['IsFemale'] +
            features['IsInactive'] * self.RISK_WEIGHTS['IsInactive'] +
            features['IsSenior'] * self.RISK_WEIGHTS['IsSenior'] +
            features['Has3PlusProducts'] * self.RISK_WEIGHTS['Has3PlusProducts'] +
            features['Senior_Inactive'] * self.RISK_WEIGHTS['Senior_Inactive']
        )

        return features

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature engineering pipeline.

        Parameters
        ----------
        X : DataFrame
            Training features.
        y : ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        self
        """
        # Build and fit preprocessor
        self.preprocessor_ = self._build_preprocessor()
        self.preprocessor_.fit(X)

        # Create custom features and fit scaler
        custom_features = self._create_custom_features(X)
        self.custom_scaler_ = StandardScaler()
        self.custom_scaler_.fit(custom_features[self.CONTINUOUS_CUSTOM])

        # Store feature names
        preprocessor_names = list(self.preprocessor_.get_feature_names_out())
        custom_names = list(custom_features.columns)
        self.feature_names_ = preprocessor_names + custom_names

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features.

        Parameters
        ----------
        X : DataFrame
            Features to transform.

        Returns
        -------
        ndarray
            Transformed feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Apply preprocessor
        X_preprocessed = self.preprocessor_.transform(X)

        # Create and scale custom features
        custom_features = self._create_custom_features(X)
        custom_features[self.CONTINUOUS_CUSTOM] = self.custom_scaler_.transform(
            custom_features[self.CONTINUOUS_CUSTOM]
        )

        # Combine
        X_combined = np.hstack([X_preprocessed, custom_features.values])

        # Handle NaN/Inf
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        return X_combined

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Fit and transform features.

        Parameters
        ----------
        X : DataFrame
            Training features.
        y : ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        ndarray
            Transformed feature matrix.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        if self.feature_names_ is None:
            raise RuntimeError("Must call fit() first")
        return self.feature_names_

    def save(self, filepath: str):
        """Save the fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before save()")
        joblib.dump(self, filepath)
        print(f"Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ChurnFeatureEngineer':
        """Load a fitted pipeline."""
        return joblib.load(filepath)

    def __repr__(self):
        return f"ChurnFeatureEngineer(random_state={self.random_state})"


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, f1_score, classification_report

    print("=" * 60)
    print("CHURN FEATURE ENGINEERING - PRODUCTION PIPELINE")
    print("=" * 60)

    # Load data
    df = pd.read_csv('/mnt/project/churn.csv')
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData: {len(y_train)} train, {len(y_test)} test")
    print(f"Imbalance: {(y_train==0).sum()}:{(y_train==1).sum()} = {(y_train==0).sum()/(y_train==1).sum():.1f}:1")

    # Feature Engineering
    print("\n--- Feature Engineering ---")
    fe = ChurnFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    print(f"Input features: {X_train.shape[1]}")
    print(f"Output features: {X_train_fe.shape[1]}")
    print(f"\nFeature names ({len(fe.get_feature_names_out())}):")
    for i, name in enumerate(fe.get_feature_names_out(), 1):
        print(f"  {i:2d}. {name}")

    # Model Training with class_weight (NO SMOTE!)
    print("\n--- Model Training (class_weight='balanced') ---")

    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_fe, y_train)
        y_pred = model.predict(X_test_fe)
        y_proba = model.predict_proba(X_test_fe)[:, 1]

        results.append({
            'Model': name,
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'F1': f1_score(y_test, y_pred)
        })
        print(f"\n{name}:")
        print(f"  ROC-AUC: {results[-1]['ROC-AUC']:.4f}")
        print(f"  F1 Score: {results[-1]['F1']:.4f}")

    # Best model details
    best = max(results, key=lambda x: x['ROC-AUC'])
    print(f"\n🏆 Best Model: {best['Model']} (ROC-AUC: {best['ROC-AUC']:.4f})")

    # Save pipeline
    print("\n--- Save Pipeline ---")
    fe.save('/home/claude/churn_feature_engineer_v2.joblib')

    # Test loading
    fe_loaded = ChurnFeatureEngineer.load('/home/claude/churn_feature_engineer_v2.joblib')
    X_test_loaded = fe_loaded.transform(X_test)
    print(f"Load test passed: {np.allclose(X_test_fe, X_test_loaded)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    ✓ Feature Engineering: 10 raw features → 28 engineered features
    ✓ Imbalance Handling: class_weight='balanced' (no SMOTE)
    ✓ Outliers (Age): RobustScaler + KBinsDiscretizer (not removed)
    ✓ Transforms: log(Age), log(Balance), RiskScore
    ✓ Pipeline: Saved as joblib for production deployment
    
    Next Steps:
    1. Hyperparameter tuning (GridSearchCV/Optuna)
    2. Threshold optimization for business metrics
    3. Model interpretation (SHAP values)
    4. Production deployment (FastAPI/Flask)
    """)
