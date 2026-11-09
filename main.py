"""Optimize a multi-output bioprocess DNN; export tables and portfolio figures.

Run `python main.py` for the complete experiment, or `python main.py --mode smoke`
for a quick integration test. All source paths are relative to this script.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch

from src.classes import FullyConnectedNeuralNetwork
from src.constants import (CV_EPOCHS, DATASET, FIGURES, MIN_EPOCHS,
                           PATIENCE, SEARCH_EPOCHS, SEED, SENSITIVITY_EPOCHS,
                           TABLES)
import src.functions as sf


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, index=False, mode='a', header=not path.exists())


def cached_ids(path, columns):
    if not path.exists():
        return set()
    return {tuple(str(v) for v in row) for row in pd.read_csv(path)[columns].itertuples(index=False, name=None)}


def new_model(config, epochs, seed, smoke=False):
    return FullyConnectedNeuralNetwork(
        **config, epochs=epochs, random_state=seed,
        patience=(5 if smoke else PATIENCE),
        min_epochs=(6 if smoke else MIN_EPOCHS),
    )


def run_search(x, y, directory, smoke):
    path = directory / 'search_results.csv'
    candidates = sf.balanced_search_candidates(SEED)
    if smoke:
        candidates = candidates[:4]
    i_train, i_valid = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    recorded = set(pd.read_csv(path).candidate_id) if path.exists() else set()
    for n, config in enumerate(candidates, 1):
        if n in recorded:
            continue
        print(f'SEARCH [{n}/{len(candidates)}] {config}', flush=True)
        try:
            model = new_model(config, 12 if smoke else SEARCH_EPOCHS, SEED + n, smoke)
            model.fit(x[i_train], y[i_train], x[i_valid], y[i_valid])
            rmse, nrmse, mean = sf.score_predictions(y[i_valid], model.predict(x[i_valid]))
            result = {**sf.serialize_config(config), 'candidate_id': n,
                      'mean_nrmse': mean, 'best_epoch': model.best_epoch_,
                      'epochs_run': model.epochs_run_, 'converged': model.converged_,
                      'fit_seconds': model.elapsed_seconds_, **sf.metrics_row(rmse, nrmse)}
        except (FloatingPointError, ValueError) as ex:
            result = {**sf.serialize_config(config), 'candidate_id': n,
                      'mean_nrmse': float('inf'), 'error': str(ex)}
        append_csv(path, result)
    frame = pd.read_csv(path)
    best = frame.loc[frame.mean_nrmse.idxmin()]
    config = sf.config_from_row(best)
    print(f'BEST: {config}; mean validation NRMSE={best.mean_nrmse:.5f}', flush=True)
    (directory / 'optimal_config.json').write_text(json.dumps(
        {**sf.serialize_config(config), 'epochs_search': 12 if smoke else SEARCH_EPOCHS,
         'epochs_cv': 12 if smoke else CV_EPOCHS,
         'epochs_sensitivity': 12 if smoke else SENSITIVITY_EPOCHS,
         'seed': SEED, 'n_search': len(candidates)}, indent=2), encoding='utf-8')
    return config


def read_optimal(directory):
    path = directory / 'optimal_config.json'
    if not path.exists():
        raise FileNotFoundError('Run search before this stage: optimal_config.json missing')
    return sf.config_from_row(json.loads(path.read_text(encoding='utf-8')))


def run_optimal_cv(x, y, config, directory, smoke):
    metrics_path = directory / 'optimal_4fold_metrics.csv'
    history_path = directory / 'optimal_4fold_loss.csv'
    recorded = set(pd.read_csv(metrics_path).fold) if metrics_path.exists() else set()
    kfold = KFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    for fold, (train_outer, test_outer) in enumerate(kfold.split(x), 1):
        if fold in recorded:
            continue
        train, valid = train_test_split(train_outer, test_size=0.10, random_state=SEED + fold)
        print(f'OPTIMAL CV fold {fold}/4', flush=True)
        model = new_model(config, 12 if smoke else CV_EPOCHS, SEED + fold, smoke)
        model.fit(x[train], y[train], x[valid], y[valid])
        rmse, nrmse, mean = sf.score_predictions(y[test_outer], model.predict(x[test_outer]))
        append_csv(metrics_path, {'fold': fold, **sf.metrics_row(rmse, nrmse),
                                  'mean_nrmse': mean, 'fit_seconds': model.elapsed_seconds_,
                                  'best_epoch': model.best_epoch_,
                                  'epochs_run': model.epochs_run_,
                                  'converged': model.converged_})
        for hist in model.history_:
            append_csv(history_path, {'fold': fold, **hist})
    sf.plot_cv_history(pd.read_csv(history_path), FIGURES / 'optimal_4fold_loss.png'
                       if not smoke else directory / 'optimal_4fold_loss.png')


def run_sensitivity(x, y, optimal, directory, smoke):
    path = directory / 'sensitivity_3fold_metrics.csv'
    recorded = cached_ids(path, ['factor', 'position', 'fold'])
    settings = sf.sensitivity_settings(optimal)
    kfold = list(KFold(n_splits=3, shuffle=True, random_state=SEED + 2).split(x))
    for factor, levels in settings.items():
        for position, value in enumerate(levels):
            config = {**optimal, factor: value}
            for fold, (train_outer, test_outer) in enumerate(kfold, 1):
                if (factor, str(position), str(fold)) in recorded:
                    continue
                tr, valid = train_test_split(train_outer, test_size=0.10,
                                             random_state=SEED + fold + 20)
                print(f'SENSITIVITY {factor} {position + 1}/{len(levels)}, fold {fold}/3', flush=True)
                try:
                    model = new_model(config, 12 if smoke else SENSITIVITY_EPOCHS,
                                      SEED + fold + 25, smoke)
                    model.fit(x[tr], y[tr], x[valid], y[valid])
                    rmse, nrmse, mean = sf.score_predictions(y[test_outer], model.predict(x[test_outer]))
                    row = {'factor': factor, 'position': position, 'value': str(value),
                           'fold': fold, 'mean_nrmse': mean,
                           'fit_seconds': model.elapsed_seconds_,
                           'best_epoch': model.best_epoch_,
                           'epochs_run': model.epochs_run_,
                           'converged': model.converged_,
                           **sf.metrics_row(rmse, nrmse)}
                except (FloatingPointError, ValueError) as ex:
                    row = {'factor': factor, 'position': position, 'value': str(value),
                           'fold': fold, 'mean_nrmse': float('inf'), 'error': str(ex)}
                append_csv(path, row)
    frame = pd.read_csv(path)
    summary = sf.summarize_metrics(frame, ['factor', 'position', 'value'])
    summary.to_csv(directory / 'sensitivity_summary.csv', index=False)
    sf.plot_sensitivity(summary, optimal, FIGURES / 'hyperparameter_sensitivity.png'
                        if not smoke else directory / 'hyperparameter_sensitivity.png')


def make_polynomial(degree, alpha):
    # Input feature centering precedes polynomial expansion, so cross-products
    # do not unnecessarily inflate; scale polynomial terms to regularize fairly.
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))


def fit_polynomial(model, x_train, y_train):
    # Targets are separately standardized on this fit's training subset.
    scaler = StandardScaler().fit(y_train)
    start = time.perf_counter()
    model.fit(x_train, scaler.transform(y_train))
    return scaler, time.perf_counter() - start


def run_polynomial(x, y, directory, smoke):
    search_path = directory / 'polynomial_search.csv'
    recorded = cached_ids(search_path, ['degree', 'alpha'])
    train, valid = train_test_split(np.arange(len(y)), test_size=.20, random_state=SEED)
    for degree in (1, 2, 3):
        for alpha in (0.01, 1.0, 100.0):
            if (str(degree), str(alpha)) in recorded:
                continue
            print(f'POLYNOMIAL SEARCH degree={degree}, alpha={alpha}', flush=True)
            model = make_polynomial(degree, alpha)
            target_scaler, elapsed = fit_polynomial(model, x[train], y[train])
            predicted = target_scaler.inverse_transform(model.predict(x[valid]))
            rmse, nrmse, mean = sf.score_predictions(y[valid], predicted)
            append_csv(search_path, {'degree': degree, 'alpha': alpha,
                                     'mean_nrmse': mean, 'fit_seconds': elapsed,
                                     **sf.metrics_row(rmse, nrmse)})
    best = pd.read_csv(search_path).sort_values('mean_nrmse').iloc[0]
    metrics_path = directory / 'polynomial_4fold_metrics.csv'
    recorded = set(pd.read_csv(metrics_path).fold) if metrics_path.exists() else set()
    for fold, (tr, te) in enumerate(KFold(n_splits=4, shuffle=True,
                                          random_state=SEED + 1).split(x), 1):
        if fold in recorded:
            continue
        print(f'POLYNOMIAL CV fold {fold}/4', flush=True)
        model = make_polynomial(int(best.degree), float(best.alpha))
        target_scaler, elapsed = fit_polynomial(model, x[tr], y[tr])
        predicted = target_scaler.inverse_transform(model.predict(x[te]))
        rmse, nrmse, mean = sf.score_predictions(y[te], predicted)
        append_csv(metrics_path, {'fold': fold, 'degree': int(best.degree),
                                  'alpha': float(best.alpha), 'fit_seconds': elapsed,
                                  'mean_nrmse': mean, **sf.metrics_row(rmse, nrmse)})


def run_comparison(directory, smoke):
    neural = pd.read_csv(directory / 'optimal_4fold_metrics.csv')
    poly = pd.read_csv(directory / 'polynomial_4fold_metrics.csv')
    neural.insert(0, 'model', 'Neural network')
    poly.insert(0, 'model', 'Polynomial regression')
    both = pd.concat((neural, poly), ignore_index=True)
    both.to_csv(directory / 'model_comparison.csv', index=False)
    sf.plot_comparison(both, FIGURES / 'model_comparison.png'
                       if not smoke else directory / 'model_comparison.png')
    print('\nFINAL NRMSE (mean over 4 folds)', flush=True)
    print(both.groupby('model')[['nrmse_growth', 'nrmse_death',
                                  'nrmse_production', 'mean_nrmse']].agg(['mean','std']).round(5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full', 'smoke'), default='full')
    parser.add_argument('--phase', choices=('all', 'search', 'cv', 'sensitivity',
                                            'polynomial', 'plots'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)  # Avoid small-model CPU thread oversubscription
    np.random.seed(SEED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    directory = TABLES if args.mode == 'full' else TABLES / 'smoke'
    directory.mkdir(parents=True, exist_ok=True)
    df, x, y = sf.load_dataset(DATASET)
    if args.mode == 'smoke':
        ix = np.random.default_rng(SEED).choice(len(y), size=1536, replace=False)
        x, y = x[ix], y[ix]
    print(f'Dataset: {len(y):,} samples, {x.shape[1]} inputs, {y.shape[1]} targets', flush=True)
    if args.phase in ('all', 'search'):
        config = run_search(x, y, directory, args.mode == 'smoke')
    else:
        config = read_optimal(directory)
    if args.phase in ('all', 'cv'):
        run_optimal_cv(x, y, config, directory, args.mode == 'smoke')
    if args.phase in ('all', 'sensitivity'):
        run_sensitivity(x, y, config, directory, args.mode == 'smoke')
    if args.phase in ('all', 'polynomial'):
        run_polynomial(x, y, directory, args.mode == 'smoke')
    if args.phase in ('all', 'plots'):
        if args.phase == 'plots':
            run_optimal_cv(x, y, config, directory, args.mode == 'smoke')
            # The previous call rebuilds the figure from cached histories.
            frame = pd.read_csv(directory / 'sensitivity_summary.csv')
            sf.plot_sensitivity(frame, config, FIGURES / 'hyperparameter_sensitivity.png'
                                if args.mode == 'full' else directory / 'hyperparameter_sensitivity.png')
        run_comparison(directory, args.mode == 'smoke')
    print('Finished. Experiment tables are in:', directory, flush=True)


if __name__ == '__main__':
    main()
