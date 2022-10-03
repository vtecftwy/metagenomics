import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_NAME = 'src'
TESTS = PROJECT_ROOT / 'tests'

# sys.path.insert(1, str(PROJECT_ROOT/PACKAGE_NAME))


from src.wandb import validate_config

def test_validate_with_missing_keys():
    # Setup
    config = {
        'architecture': 'cnn-virus-original',
    }

    true_config = {
        'architecture': 'cnn-virus-original',
        'dataset': 'Dataset.map(string_to_tensor) v2',
        'n_train_samples': 0,
        'n_val_samples': 0,
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'epochs': 5,
        'ds_cache': False,
        'ds_prefetch': True,
    }

    # Execute
    result = validate_config(config)

    # Verify
    assert result == true_config

def test_validate_with_missing_and_additional_keys():
    # Setup
    config = {
        'architecture': 'cnn-virus-original',
        'new_key1': 12345,
        'new_key2': 2345,
        'new_key3': 345,
    }

    true_config = {
        'architecture': 'cnn-virus-original',
        'new_key1': 12345,
        'new_key2': 2345,
        'new_key3': 345,
        'dataset': 'Dataset.map(string_to_tensor) v2',
        'n_train_samples': 0,
        'n_val_samples': 0,
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'epochs': 5,
        'ds_cache': False,
        'ds_prefetch': True,
    }

    # Execute
    result = validate_config(config)

    # Verify
    assert result == true_config

def test_validate_with_no_keys():
    # Setup
    config = {}

    true_config = {
        'architecture': 'cnn-virus-original',
        'dataset': 'Dataset.map(string_to_tensor) v2',
        'n_train_samples': 0,
        'n_val_samples': 0,
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'epochs': 5,
        'ds_cache': False,
        'ds_prefetch': True,
    }

    # Execute
    result = validate_config(config)

    # Verify
    assert result == true_config


if __name__ == '__main__':

    print('END')