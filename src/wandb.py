# Classes and functions to support WandB process
# COMMENT: What to do in this module?
# [ ]   Define main operations involving WandB
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
import pandas as pd
import wandb


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
    projects = projects(entity)
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
    cols = 'run_name run_id at_name at_type at_id at_state at_version at_aliases file_count created_at updated_at'.split(' ')
    artifacts_df = pd.DataFrame(columns=cols)
    
    for r in runs:
        r_name, r_id = r.name, r.id
        for at in r.logged_artifacts():
            metadata = [r_name, r_id, at.name, at.type, at.id, at.state, at.version, at.aliases, at.file_count, at.created_at, at.updated_at]
            row = pd.DataFrame({k:v for k, v in zip(cols, metadata)})
            artifacts_df = artifacts_df.append(row)

    cols2show = 'run_name run_id at_name at_version at_type at_aliases file_count created_at updated_at at_id'.split(' ')

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


