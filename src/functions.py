"""Dataset preparation, model scoring, experiment specifications and plots."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.constants import FEATURES, TARGETS, TARGET_SHORT, SEARCH_SPACE, N_SEARCH


def load_dataset(path):
    df = pd.read_csv(path)
    expected = list(TARGETS) + list(FEATURES)
    missing = [c for c in expected if c not in df]
    if missing:
        raise ValueError(f'Dataset is missing required columns: {missing}')
    if df[expected].isna().any().any():
        raise ValueError('Missing values are not supported in this dataset')
    if not np.isfinite(df[expected].to_numpy(dtype=float)).all():
        raise ValueError('The dataset contains infinite or nonnumeric values')
    return df, df[list(FEATURES)].to_numpy(), df[list(TARGETS)].to_numpy()


def score_predictions(y_true, y_pred):
    """Per-target RMSE, NRMSE=RMSE/IQR of held-out data, and mean NRMSE."""
    rmse = np.asarray(root_mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    iqr = np.percentile(y_true, 75, axis=0) - np.percentile(y_true, 25, axis=0)
    if np.any(iqr <= 0):
        raise ValueError('A target has zero IQR in the scoring set')
    nrmse = rmse / iqr
    return rmse, nrmse, float(nrmse.mean())


def settings_id(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def config_from_row(row):
    return {
        'hidden_layers': tuple(json.loads(row['hidden_layers'])),
        'activation': str(row['activation']),
        'learning_rate': float(row['learning_rate']),
        'batch_size': int(row['batch_size']),
        'weight_decay': float(row['weight_decay']),
    }


def serialize_config(config):
    out = config.copy()
    out['hidden_layers'] = json.dumps(list(config['hidden_layers']))
    return out


def balanced_search_candidates(seed):
    """24 unique, reproducible candidates with equal counts at every factor level.

    The full factorial design has 3*4*3*3*3=324 combinations. Building each
    column from repeated and separately shuffled levels guarantees balance.
    """
    rng = np.random.default_rng(seed)
    names = list(SEARCH_SPACE)
    for _ in range(10000):
        columns = []
        for name in names:
            values = SEARCH_SPACE[name]
            assert N_SEARCH % len(values) == 0
            candidates = list(values) * (N_SEARCH // len(values))
            rng.shuffle(candidates)
            columns.append(candidates)
        configs = [dict(zip(names, values)) for values in zip(*columns)]
        if len({settings_id(x) for x in configs}) == N_SEARCH:
            return configs
    raise RuntimeError('Unable to produce balanced unique search')


def sensitivity_settings(optimal):
    """For each factor, compare baseline with two smaller and two larger values."""
    configs = {}
    architecture_ladder = [(4, 2), (8, 4), (16, 8), (32, 16),
                           (64, 32), (128, 64), (256, 128)]
    index = architecture_ladder.index(tuple(optimal['hidden_layers']))
    configs['hidden_layers'] = architecture_ladder[index - 2:index + 3]
    configs['activation'] = ['sigmoid', 'tanh', 'relu', 'leakyrelu']
    lr = optimal['learning_rate']
    configs['learning_rate'] = [lr * z for z in (0.25, 0.5, 1, 2, 4)]
    bs = optimal['batch_size']
    configs['batch_size'] = [bs // 4, bs // 2, bs, bs * 2, bs * 4]
    decay = optimal['weight_decay']
    if decay == 0:
        configs['weight_decay'] = [0.0, 0.00001, 0.0001, 0.001, 0.01]
    else:
        configs['weight_decay'] = [decay * z for z in (0.25, 0.5, 1, 2, 4)]
    return configs


def summarize_metrics(data, group_cols):
    rows = []
    for group, part in data.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group, tuple):
            group = (group,)
        row = dict(zip(group_cols, group))
        for col in ['mean_nrmse', 'rmse_growth', 'rmse_death', 'rmse_production',
                    'nrmse_growth', 'nrmse_death', 'nrmse_production', 'fit_seconds']:
            if col in part:
                row[f'{col}_mean'] = float(part[col].mean())
                row[f'{col}_std'] = float(part[col].std(ddof=1)) if len(part) > 1 else 0.
        row['n_folds'] = len(part)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_row(rmse, nrmse):
    out = {}
    for i, suffix in enumerate(['growth', 'death', 'production']):
        out[f'rmse_{suffix}'] = float(rmse[i])
        out[f'nrmse_{suffix}'] = float(nrmse[i])
    return out


def plot_cv_history(loss_df, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, (fold, frame) in zip(axes.ravel(), loss_df.groupby('fold', sort=True)):
        frame = frame.sort_values('epoch')
        ax.plot(frame.epoch, frame.loss_train, label='Training loss', lw=1.6)
        ax.plot(frame.epoch, frame.loss_valid, label='Validation loss', lw=1.6)
        best = frame.loc[frame.loss_valid.idxmin()]
        ax.axvline(best.epoch, ls='--', lw=1.0, color='gray',
                   label=f'Best epoch: {int(best.epoch)}')
        ax.set_title(f'Fold {int(fold)} — {len(frame)} epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean squared error (standardized targets)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Four-fold cross-validation: training and inner-validation loss', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(summary_df, optimal, path):
    factors = ['hidden_layers', 'activation', 'learning_rate',
               'batch_size', 'weight_decay']
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for ax, name in zip(axes.ravel(), factors):
        sub = summary_df[summary_df['factor'] == name].copy()
        labels = sub['value'].tolist()
        values = sub['mean_nrmse_mean'].to_numpy()
        spread = sub['mean_nrmse_std'].to_numpy()
        xs = np.arange(len(labels))
        ax.errorbar(xs, values, yerr=spread, fmt='o-', capsize=3, lw=1.5)
        ax.set_xticks(xs, labels, rotation=30 if name == 'hidden_layers' else 0)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_ylabel('Mean NRMSE across targets (3-fold CV)')
        ax.set_xlabel('Value')
        ax.grid(alpha=.2)
    axes.ravel()[-1].axis('off')
    fig.suptitle('One-factor-at-a-time hyperparameter sensitivity', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(comparison_df, path):
    data = comparison_df[comparison_df['fold'] != 'mean'].copy()
    groups = ['Neural network', 'Polynomial regression']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(3)
    width = 0.32
    for i, model in enumerate(groups):
        subset = data[data.model == model]
        means = [subset[f'nrmse_{s}'].mean() for s in ('growth', 'death', 'production')]
        stds = [subset[f'nrmse_{s}'].std(ddof=1) for s in ('growth', 'death', 'production')]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
               capsize=3, label=model)
    ax.set_xticks(x, TARGET_SHORT)
    ax.set_ylabel('NRMSE = RMSE / held-out IQR (mean ± SD of 4 folds)')
    ax.set_title('Out-of-fold error: optimized neural network vs polynomial regression')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
