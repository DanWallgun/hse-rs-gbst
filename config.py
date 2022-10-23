import importlib
import yaml
import torch
from torch.utils.data import DataLoader
from data import load_dataset, collate_batch


def _init_obj(module_name, obj_name, obj_config, **kwargs):
    obj_config.update(kwargs)
    return getattr(importlib.import_module(module_name), obj_name)(**obj_config)

def _load_setup(config_name):
    config = yaml.safe_load(open(config_name))
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    text_encoder = _init_obj('data', config['text_encoder'].pop('class'), config['text_encoder'])
    
    model = _init_obj('models', config['model'].pop('class'), config['model'], ntoken=len(text_encoder))
    model.to(device)
    
    ds = load_dataset(config['data']['name'])
    collator = lambda batch: collate_batch(batch, text_encoder, device)
    train_dl = DataLoader(
        ds['train'],
        batch_size=config['data']['bsz'],
        shuffle=True,
        collate_fn=collator
    )
    valid_dl = DataLoader(
        ds['test'],
        batch_size=config['data']['bsz'],
        shuffle=False,
        collate_fn=collator
    )
    return text_encoder, model, train_dl, valid_dl


def get_chars_setup():
    return _load_setup('configs/chars.yaml')


def get_bpe_setup():
    return _load_setup('configs/bpe.yaml')


def get_gbst_setup():
    return _load_setup('configs/gbst.yaml')