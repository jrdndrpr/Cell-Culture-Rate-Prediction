"""Configuration for reproducible cell-culture rate prediction experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'datasets' / 'dataset_cell_growth_7factor.csv'
FIGURES = ROOT / 'figures'
TABLES = ROOT / 'tables'

FEATURES = (
    'Temperature [C]',
    'pH',
    'Glucose [mM]',
    'Glutamine [mM]',
    'Lactate [mM]',
    'Ammonia [mM]',
    'Viable Cell Density [cells/mL]',
)
TARGETS = (
    'Specific Growth Rate [d^-1]',
    'Specific Death Rate [d^-1]',
    'Specific Production Rate [pg cell^-1 d^-1]',
)
TARGET_SHORT = ('Growth rate', 'Death rate', 'Production rate')
SEED = 4360

# A balanced sample of 24 candidates from a 324-combination design.
SEARCH_SPACE = {
    'hidden_layers': ((16, 8), (32, 16), (64, 32)),
    'activation': ('sigmoid', 'tanh', 'relu', 'leakyrelu'),
    'learning_rate': (0.0005, 0.001, 0.002),
    'batch_size': (128, 256, 512),
    'weight_decay': (0.0, 0.001, 0.01),
}
N_SEARCH = 24
SEARCH_EPOCHS = 90
CV_EPOCHS = 260
SENSITIVITY_EPOCHS = 110
PATIENCE = 18
MIN_EPOCHS = 35
