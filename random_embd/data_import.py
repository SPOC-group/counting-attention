import torch
from pathlib import Path
import pandas as pd
import wandb
import numpy as np

from random_embd.memory_exp import TransformerSeq2Seq, Transformer2Layer

CONFIG_COLS = ['T', 'model_dim', 'p', 'seq_len', 'attention_input', 'no_softmax', 'dataset_type']
FIGURE_DIR = Path('figures')

api = wandb.Api()

def print_specs(model_spec):
    print(' | '.join([f"{k}: {v}" for k,v in model_spec.items() if k in CONFIG_COLS]))

def load_result_table(project="feeds/v4_dvsT"):
    # Project is specified by <entity/project-name>
    runs = api.runs(project)

    summary_list = []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        res = {**run.summary._json_dict,
            **{k: v for k,v in run.config.items()
            },'name':run.name,'entity': run.entity, 'project': run.project}
        #config_cols = list(run.config.keys())
        summary_list.append(res)

    df = pd.DataFrame(summary_list)
    return df.set_index('name') #, config_cols

def load_model(df_row):
    model_spec = df_row
    project = model_spec['project']
    run_name = model_spec.name
    entity = model_spec['entity']
    has_BOS = model_spec['dataset_type'] == 'backward_BOS'
    
    print(f"{entity}/{project}/{run_name}-model:v0 | val_acc: {model_spec['val_acc']:.3f}")
    model_file = api.artifact(f"{entity}/{project}/{run_name}-model:v0")
    model_path = Path(model_file.download())
    device = 'cuda:0'

    config = model_spec

    has_BOS = config['dataset_type'] == 'backward_BOS'
    
    # check if key model_type is in config
    if 'model_type' not in config:
        config['model_type'] = '1layer'
        
    model_cls = {'1layer': TransformerSeq2Seq,
                    '2layer': Transformer2Layer}[config['model_type']]
    model = model_cls(T=config['T']+int(has_BOS),
                                                model_dim=config['model_dim'],
                                                p=config['p'],n_classes=config['seq_len']+1,
                                                L=config['seq_len'],
                                                attention_input=config['attention_input'],
                                                
                                            is_embedded=True,
                                            
                                                use_softmax=not config['no_softmax']).to(device)

    model.load_state_dict(torch.load(model_path / 'transformer.pt'))
    
    return model


MODELS = {
    'linear': {'attention_input': 'linear', 'no_softmax': True, 'dataset': 'backward'},
    'linear+sftm': {'attention_input': 'linear', 'no_softmax': False, 'dataset': 'backward'},
    'dot+sftm': {'attention_input': 'only_sem', 'no_softmax': False, 'dataset': 'backward'},
    'dot': {'attention_input': 'only_sem', 'no_softmax': True, 'dataset': 'backward'},
    'dotBOS+sftm': {'attention_input': 'only_sem', 'no_softmax': False, 'dataset': 'backward_BOS'},
    'dotBOS': {'attention_input': 'only_sem', 'no_softmax': True, 'dataset': 'backward_BOS'},
}

def to_cmd(model):
    return f" --attention_input={model['attention_input']} {'--no_softmax' if model['no_softmax'] else ''} --dataset_type={model['dataset']}"