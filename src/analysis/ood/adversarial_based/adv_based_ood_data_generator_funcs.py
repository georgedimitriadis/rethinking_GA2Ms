
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from cma import CMAEvolutionStrategy
from tqdm import tqdm
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ec.elco import ECRegressor
from sklearn.linear_model import RidgeCV, LassoLarsCV
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import NuSVR
from sklearn.pipeline import Pipeline

from analysis.utils import funcs_data_manipulation as dmf


# ---------------------------
# Normalising data for rectangle search
# ---------------------------
def data_norm(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    return X_norm

# ---------------------------
# Baseline
# ---------------------------
def baseline_split(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-100)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return mean_squared_error(yte, pred), r2_score(yte, pred)


# ---------------------------
# Permutation evaluator
# ---------------------------
def evaluate_perm(perm, X, y, train_size):
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    model = LGBMRegressor(n_estimators=200, verbose=-100)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    return mean_squared_error(y[test_idx], pred), r2_score(y[test_idx], pred)


# ---------------------------
# IsolationForest adversarial split
# ---------------------------
def isolation_forest_split(X, y, train_size):
    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42)
    iso.fit(X)
    normality_scores = iso.decision_function(X)
    anomaly_scores = -normality_scores
    order = np.argsort(anomaly_scores)
    train_idx = order[:train_size]
    test_idx = order[train_size:]
    model = LGBMRegressor(n_estimators=200, verbose=-100)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    return order, mean_squared_error(y[test_idx], pred), r2_score(y[test_idx], pred)


# ---------------------------
# CMA permutation search
# ---------------------------
def cma_es_search(X, y, train_size, iters):
    n = len(X)
    es = CMAEvolutionStrategy(n * [0.0], 0.5, {'seed': 42, 'CMA_diagonal': True, 'popsize': 16})
    best_perm = None
    best_mse = -np.inf
    best_r2 = None
    pbar = tqdm(range(iters), desc="CMA perm worst=???")
    for _ in pbar:
        candidates = es.ask()
        losses = []
        for x in candidates:
            order = np.argsort(x)
            mse, r2 = evaluate_perm(order, X, y, train_size)
            losses.append(-mse)    # CMA minimizes; we want to maximize mse
            if mse > best_mse:
                best_mse = mse
                best_r2 = r2
                best_perm = order.copy()
        es.tell(candidates, losses)
        pbar.set_description(f"CMA perm worst={best_mse:.3f}")
    return best_perm, best_mse, best_r2


# ---------------------------
# Rectangle evaluation
# ---------------------------

def percentile_rank(value, reference):
    """
    value: scalar
    reference: 1D array
    returns percentile in [0, 100]
    """
    ref_sorted = np.sort(reference)
    return 100.0 * np.searchsorted(ref_sorted, value, side="right") / len(ref_sorted)

def evaluate_rectangle(z, X_norm, X_raw, y, k_ratio=0.9, use_knr=True):
    """
        z           : optimizer vector (length 2*d)
        X_norm      : normalized features in [0,1], shape (n, d)
        X_raw       : raw features, shape (n, d)
        y           : targets
        k_ratio     : percentage of selected points
    """
    n = len(X_raw)
    train_size = int(k_ratio * n)
    k = n - train_size

    d = X_norm.shape[1]
    margin = 0.02

    # ------------------------------------------------------------
    # 1. Map z → smooth latent box (USED ONLY FOR RANKING)
    # ------------------------------------------------------------
    raw_coords = margin + (1.0 - 2 * margin) / (1.0 + np.exp(-z))
    L_search = np.minimum(raw_coords[:d], raw_coords[d:2 * d])
    U_search = np.maximum(raw_coords[:d], raw_coords[d:2 * d])

    # ------------------------------------------------------------
    # 2. Distance to box (Chebyshev distance)
    # ------------------------------------------------------------
    dif_low = np.maximum(L_search - X_norm, 0.0)
    dif_high = np.maximum(X_norm - U_search, 0.0)
    dist_to_box = np.max(np.maximum(dif_low, dif_high), axis=1)

    # ------------------------------------------------------------
    # 3. ACTUAL selection (exactly k points)
    # ------------------------------------------------------------
    order = np.argsort(dist_to_box)
    test_idx = order[:k]
    train_idx = order[k:]

    # ------------------------------------------------------------
    # 4. REPORTING FROM ACTUAL SELECTED POINTS (NOT THE BOX)
    # ------------------------------------------------------------
    X_sel = X_norm[test_idx]  # <-- THIS is the truth

    L_report = np.zeros(d)
    U_report = np.zeros(d)

    for j in range(d):
        L_report[j] = percentile_rank(
            X_sel[:, j].min(),
            X_norm[:, j]
        )
        U_report[j] = percentile_rank(
            X_sel[:, j].max(),
            X_norm[:, j]
        )

    # ------------------------------------------------------------
    # 5. Penalties
    # ------------------------------------------------------------
    if len(train_idx) < 1 or len(test_idx) < k:
        return 1e10, -1e9, (L_report, U_report), test_idx

    # ------------------------------------------------------------
    # 6. Train / evaluate model
    # ------------------------------------------------------------
    if use_knr:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import GridSearchCV
        # Scaling is highly recommended for KNN to normalize distance calculations
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])

        # 3. Define the parameter grid to search
        # Note: Use 'knn__' prefix to target the named step in the pipeline
        param_grid = {
            'knn__n_neighbors': [1, 3, 5, 7, 9, 11],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        }

        # 4. Initialize GridSearchCV
        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,  # 5-fold Cross-Validation
            scoring='neg_mean_squared_error',  # Sklearn maximizes score, so we use negative MSE
            n_jobs=-1,  # Use all available CPU cores
            verbose=0
        )
    else:
        clf = Pipeline([
             ('scaler', StandardScaler()),
             ('nusvm', NuSVR())
         ])

    try:
        clf.fit(X_raw[train_idx], y[train_idx])
        pred = clf.predict(X_raw[test_idx])

        mse = mean_squared_error(y[test_idx], pred)
        r2 = r2_score(y[test_idx], pred)

        return mse, r2, (L_report, U_report), test_idx

    except Exception:
        return 1e10, -1e9, (L_report, U_report), test_idx


# ---------------------------
# CMA rectangle search
# ---------------------------
def cma_rectangle_search(X_norm, X_raw, y, k_ratio=0.9, iters=100, max_num_of_mse_drops=-1, return_full_history=False,
                         dataset_name=None, use_knr=True):
    d = X_norm.shape[1]
    L0 = 0.2 * np.ones(d)
    U0 = 0.8 * np.ones(d)
    def inv_sig(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    z0 = np.concatenate([inv_sig(L0), inv_sig(U0)])
    sigma0 = 0.5
    es = CMAEvolutionStrategy(list(z0), sigma0, {
                                                # 'CMA_diagonal': True,
                                                 'popsize': 20,
                                                 'verbose': -8})

    best_mse = -np.inf
    best_r2 = None
    best_bounds = None
    best_test_idx = None

    all_unique_results = []
    tqdm_description = 'Rect worst=???' if dataset_name is None else dataset_name
    pbar = tqdm(range(iters), desc=tqdm_description)
    for idx in pbar:
        candidates = es.ask()
        losses = []
        for z in candidates:
            if not use_knr:
                y = y.ravel()
            mse, r2, bounds, test_idx = evaluate_rectangle(z, X_norm, X_raw, y, k_ratio=k_ratio, use_knr=use_knr)
            losses.append(-mse)  # ← FIXED: Negate MSE since CMA minimizes
            if mse > best_mse:
                best_mse = mse
                best_r2 = r2
                best_bounds = bounds
                best_test_idx = test_idx.copy()
        if len(all_unique_results) == 0 or best_mse > all_unique_results[-1][1]:
            all_unique_results.append((idx, best_mse, best_r2, best_bounds, best_test_idx))
        es.tell(candidates, losses)
        tqdm_description = f'Rect worst={best_mse:.3f}' if dataset_name is None else f'{dataset_name} Rect worst = {best_mse:.3f}'
        pbar.set_description(tqdm_description)
        if return_full_history and len(all_unique_results) == max_num_of_mse_drops:
            break
        #if return_full_history and len(all_unique_results) <= 2 and idx > 100:
        #    return None
    if not return_full_history:
        return best_mse, best_r2, best_bounds, best_test_idx
    else:
        return all_unique_results


# ---------------------------
# Evaluate other regressors
# ---------------------------
def evaluate_other_models(X, y, train_idx, test_idx):
    results = {}

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X[train_idx], y[train_idx])
    pred_rf = rf.predict(X[test_idx])
    results['RandomForest'] = {
        'MSE': mean_squared_error(y[test_idx], pred_rf),
        'R2': r2_score(y[test_idx], pred_rf)
    }

    # NuSVR
    nu_svr = Pipeline([
        ('scaler', StandardScaler()),
        ('nusvm', NuSVR(
            nu=0.5,  # Controls the number of support vectors (0 < nu <= 1)
            C=1.0,  # Regularization parameter
            kernel='rbf',  # RBF kernel is generally a good default
            gamma='scale',  # Kernel coefficient
            cache_size=200  # Memory cache size in MB
        ))
    ])
    nu_svr.fit(X[train_idx], y[train_idx])
    pred_svr = nu_svr.predict(X[test_idx])
    results['NuSVR'] = {
        'MSE': mean_squared_error(y[test_idx], pred_svr),
        'R2': r2_score(y[test_idx], pred_svr)
    }

    # GradientBoosting
    gbr = LGBMRegressor(n_estimators=2000, random_state=42)
    gbr.fit(X[train_idx], y[train_idx])
    pred_gbr = gbr.predict(X[test_idx])
    results['GradientBoosting'] = {
        'MSE': mean_squared_error(y[test_idx], pred_gbr),
        'R2': r2_score(y[test_idx], pred_gbr)
    }

    train_idx_split, val_idx_split = train_test_split(train_idx, test_size=0.1, random_state=42)

    train = X[train_idx_split]
    val = X[val_idx_split]

    test_fold = np.concatenate([
        np.full(len(train), -1),  # Training samples
        np.zeros(len(val))  # Validation samples
    ])

    # Create the predefined split
    ps = PredefinedSplit(test_fold)

    ec = ECRegressor(epochs=5000,
                     learning_rate=0.001,
                     batch_size=16,
                     validation_split=0.1,
                     final_linear_layer_regularizer=None,
                     arity=2,
                     ps=ps
                     )

    ec = Pipeline([
        ("scaler", MinMaxScaler((-1, 1))),
        ('regressorclassifier', ec),
        ("final", RidgeCV(cv=ps))

    ])

    ec.fit(X[train_idx], y[train_idx])
    pred_ec = ec.predict(X[test_idx])
    results['EC'] = {
        'MSE': mean_squared_error(y[test_idx], pred_ec),
        'R2': r2_score(y[test_idx], pred_ec)
    }

    return results




