import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

CATEGORICAL_COLS = ["category", "gender", "job"]
COLS_TO_SCALE = ["amt", "city_pop", "age", "distance_km", "hour"]

def encode_categorical_features(X_train: pd.DataFrame, X_recent: pd.DataFrame, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    
    encoders_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        encoders_dict[col] = le
        
        # Aplicar el fallback seguro al subset reciente
        X_recent[col] = X_recent[col].astype(str).apply(
            lambda v: v if v in le.classes_ else le.classes_[0]
        )
        X_recent[col] = le.transform(X_recent[col])
        
    return X_train, X_recent, encoders_dict

def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame, X_recent: pd.DataFrame, cols_to_scale=None):
    if cols_to_scale is None:
        cols_to_scale = COLS_TO_SCALE
        
    scaler = RobustScaler()
    X_train[cols_to_scale]  = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]   = scaler.transform(X_test[cols_to_scale])
    
    X_recent_scaled = X_recent.copy()
    X_recent_scaled[cols_to_scale] = scaler.transform(X_recent[cols_to_scale])
    
    return X_train, X_test, X_recent_scaled, scaler