# Classes and functions to support WandB process
# COMMENT: What to do in this module?
# []   Define main operations involving WandB
#       - preparing dataset and loading datasets as artifact
#       - training a model with a dataset and hyperparameters
#       - testing / validating model with test dataset
#       - infering with now data
# 
# 
# [ ]   Function required in dataset preparation -> raw_data, processed_data, code, 
# 
# [ ]   Function required in model training -> dataset, code, model, hyperparameters
#       - how to track epochs, learning rates and other 
#       - how to track gradients and activations
# 
# [ ]   Function required in model testing -> dataset, code, trained model, preformance and reports

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import wandb

from datetime import datetime
from pathlib import Path
from src.architecture import build_model, build_cnn_virus_original
from src.datasets import strings_to_tensors
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from typing import Callable
from wandb.keras import WandbCallback


def register_code_to_wandb(code_fname, p2drive):
    """Register code file name to WandB to allow it to be logged in runs"""
    p2nb = p2drive / 'nbs' / code_fname
    if p2nb.is_file():
        os.environ['WANDB_NOTEBOOK_NAME'] = str(p2nb.absolute())
    else:
        raise ValueError(f"code file {p2nb.name} does not exist")

    print(f"{'>'*0} Verify that path and name to the code are correct: {'<'*25}")
    print(f" - file name:     {code_fname}")
    print(f" - file path:     {os.environ['WANDB_NOTEBOOK_NAME']}")

def get_project(entity, project_name):
    """Returns WandB project object defined by entity and project name"""
    api = wandb.Api()
    return api.from_path(f"{entity}/{project_name}")

def entity_projects(entity):
    """Returns wandb.apis.public.Projects object, iterable collection of all projects in entity"""
    api = wandb.Api()
    projects = api.projects(entity=entity)
    return projects

def print_entity_project_list(entity):
    """Print the name and url of all projects in entity"""
    projects = entity_projects(entity)
    print(f"List of projects under entity <{entity}>")
    for i, p in enumerate(projects):
        print(f" {i:2d}. {p.name:30s} (url: {p.url})")

def project_artifacts(entity, project_name, by_alias='latest', by_type=None, by_version=None):
    """Returns a tuple (artifacts df, list) with key information on project artifacts, filtered by alias and types:
    
    artifacts_df:   pd.DataFrame listing artifacts in the project. The list of artifacts is filtered by their aliases:
                    by_alias='latest':      only returns the latest version of each artifact (default)
                    by_alias=None:          returns artifacts with any alias
                    by_type/by_version=None:returns artifacts with any type/version (default)
                    by_type/by_version=str: only returns the artifacts with the passed type/version
    """

    api = wandb.Api()
    project = api.from_path(f"{entity}/{project_name}")
    at_types = project.artifacts_types()
    runs = api.runs(path=f"{entity}/{project_name}")

    # validate by_type parameter
    if by_type is not None and by_type not in [t.name for t in at_types]:
        raise ValueError(f"{by_type} is not an artifact type in {entity}/{project_name}")

    # create a df where each row corresponds to one artifact logged during one run in this project
    # some artifact may be duplicated when linked to more than one run. Those duplicate need to be filtered out
    cols = 'at_name at_type at_id at_state at_version at_aliases file_count created_at updated_at'.split(' ')
    artifacts_df = pd.DataFrame(columns=cols)
    
    for r in runs:
        for at in r.logged_artifacts():
            metadata = [at.name, at.type, at.id, at.state, at.version, at.aliases, at.file_count, at.created_at, at.updated_at]
            row = pd.DataFrame({k:v for k, v in zip(cols, metadata)})
            artifacts_df = artifacts_df.append(row)
    artifacts_df = artifacts_df.loc[~artifacts_df.duplicated(subset=['at_id'], keep='first'), :]

    cols2show = 'at_name at_version at_type at_aliases file_count created_at updated_at at_id'.split(' ')
    # filtering by passed alias and type:
    #   if by_xxx is not None:    filter is a boolean vector
    #   if by_xxx is None:        filter is an array of 'True'
    nbr_rows = artifacts_df.shape[0]
    alias_filter = artifacts_df.at_aliases==by_alias if by_alias is not None else np.ones(shape=(nbr_rows,), dtype=bool)
    type_filter = artifacts_df.at_type==by_type if by_type is not None else np.ones(shape=(nbr_rows,), dtype=bool)
    version_filter = artifacts_df.at_version==by_version if by_version is not None else np.ones(shape=(nbr_rows,), dtype=bool)

    row_filter = alias_filter * type_filter * version_filter

    latest = artifacts_df.loc[row_filter, cols2show].sort_values(by='created_at').reset_index(drop=True)
    return latest, [t.name for t in at_types]

def run_exists(run_name, entity, project):
    """Return True if there is already a run with the same name in the given entity and project"""
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}")
    run_matches = [run_name == r.name for r in runs]
    return any(run_matches)

def unique_run_name(name_seed):
    timestamp = datetime.now().strftime('%y%m%d-%H%M')
    return f"{name_seed}-{timestamp}"

def validate_config(config):
    """returns config where missing required keys are replaced into with default values"""
    default_config = {
        'architecture': 'cnn-virus-original',
        'dataset': 'Dataset.map(string_to_tensor) v2',
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'epochs': 5,
        'ds_cache': False,
        'ds_prefetch': True,
    }

    if config is None: config = {}

    for k, v in default_config.items():
        config[k] = config.get(k, default_config[k])
    
    return config

def train_with_wandb(
    entity: str, project_name: str, run_name_seed: str, 
    train_ds_at_name: str, val_ds_at_name: str, 
    model_at_name: str = None, build_model: Callable = None,
    config: dict = None 
    ):
    """Starts a new wandb run and performs a training sequence using datasets and (optional) saved model.
    
    The function perform each of these steps:
        1. validate the config
        2. starts a new wandb run using the run_name_seed and the config dictionary
        3. downloads train and val raw text datasets and transforms them with the transform function
        4. load the selected saved model or creates new model, and compile it
        5. train the model, using wandb to track and save intermediate models
        6. finish the run

    Parameters
    ----------
    entity : str
        name of the WandB user or organization to use to create the new run
    project_name : str
        name of the project to use to create the new run. 
    run_name_seed : str
        name to give to the new run
        the run display name will be this string followed by a timestamp
    train_ds_at_name : str
        name of the WandB Artifact with the train dataset.
        the name should not include any `:vn` version number
    val_ds_at_name : str
        name of the WandB Artifact with the validation dataset.
        the name should not include any `:vn` version number
    config : dict
        dictionary where key-value pairs represent all the metadata to store with the run
        the key-value pairs below are required and will be set as the default values if not present:
            'architecture' (default: 'cnn-virus-original')
            'dataset': (default: 'Dataset.map(string_to_tensor) v2')
            'n_train_samples': (default: 0) (retrieved from artifact metadata if available)
            'n_val_samples': (default: 0)   (retrieved from artifact metadata if available)
            'batch_size': (default: 1024)
            'learning_rate': (default: 1e-3)
            'epochs': (default: 5)
            'ds_cache': (default: False)
            'ds_prefetch': (default: True)
    model_at_name : str, default=None
        name of the WandB Artifact with the saved moded to use.
        the name should not include any `:vn` version number
        when `None`, a new model is created
    build_model : Callable,
        function to build an empty architecture

    """
    run_name = unique_run_name(run_name_seed)

    # 1. validate configuration
    config = validate_config(config)

    # Retrieve n_samples from dataset artifacts metadata and save in config
    train_ds_at_path = f"{entity}/{project_name}/{train_ds_at_name}:latest"
    val_ds_at_path =   f"{entity}/{project_name}/{val_ds_at_name}:latest"
    api = wandb.Api()
    train_at = api.artifact(train_ds_at_path)
    val_at = api.artifact(val_ds_at_path)
    config['n_train_samples'] = train_at.metadata.get('n_samples', 0)
    config['n_val_samples'] = val_at.metadata.get('n_samples', 0)

    # 2. start a new run
    run = wandb.init(
        entity=entity, 
        project=project_name, 
        name=run_name, job_type="train-exp", 
        config=config, 
        save_code=True
        )
    cfg = wandb.config

    # 3a. download train and val raw data files

    train_ds_at = run.use_artifact(train_ds_at_path, type='raw_data')
    train_ds_dir = train_ds_at.download()
    train_ds_file = list(Path(train_ds_dir).iterdir())[0]

    val_ds_at = run.use_artifact(val_ds_at_path, type='raw_data')
    val_ds_dir = val_ds_at.download()
    val_ds_file = list(Path(val_ds_dir).iterdir())[0]

    print(f"Build Datasets from files {train_ds_file.name} and {val_ds_file.name}")

    # 3.b create Datasets for train and val
    text_train_ds = tf.data.TextLineDataset(
        train_ds_file,
        compression_type='',
        name='text_train_ds'
    ).batch(cfg['batch_size'])

    text_val_ds = tf.data.TextLineDataset(
        val_ds_file,
        compression_type='',
        name='text_val_ds'
    ).batch(cfg['batch_size'])

    if config['ds_cache'] and config['ds_prefetch']:
        train_ds = text_train_ds.map(strings_to_tensors).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = text_val_ds.map(strings_to_tensors).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    elif not config['ds_cache'] and config['ds_prefetch']:
        train_ds = text_train_ds.map(strings_to_tensors).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = text_val_ds.map(strings_to_tensors).prefetch(buffer_size=tf.data.AUTOTUNE)

    else:
        train_ds = text_train_ds.map(strings_to_tensors)
        val_ds = text_val_ds.map(strings_to_tensors)

    print(f"dataset built with cache:{config['ds_cache']}, prefetch:{config['ds_prefetch']}.")

    # create model using passed build function or loaded artifact, and compile it
    if model_at_name is None:
        if build_model is not None and callable(build_model):
            print('Creating a new model')
            model = build_cnn_virus_original()
        else:
            raise ValueError(f"Require 'build_model' to be a callable to create a new model")
    else:
        print(f"Downloading and using latest version of model {model_at_name}")
        model_at_path = f"{project_name}/{model_at_name}:latest"
        model_at = run.use_artifact(model_at_path, type='model')
        model_at_dir = model_at.download()
        model = tf.keras.models.load_model(Path(model_at_dir).resolve())

    optim = Adam(learning_rate=wandb.config.learning_rate)
    model.compile(
        optimizer=optim,
        loss=[CategoricalCrossentropy(name='l1'), CategoricalCrossentropy(name='l2')],
        metrics=['acc']
    )
    
    # train model
    wb = WandbCallback(
        monitor=cfg['metric_to_monitor'], 
        save_model=True, 
        log_weigths=True
        )

    res = model.fit(
        train_ds,
        epochs=wandb.config.epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[wb]
        )

    run.finish()    

if __name__ == '__main__':
    pass