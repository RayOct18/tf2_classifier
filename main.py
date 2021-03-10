from trainer import Model
from data import DataLoader
from utils import gpu_select, load_config


def main(exp):
    gpu_select()
    cfg = load_config(f'{exp}.yaml')

    data_cfg = cfg['DATA']
    train_data = DataLoader(**data_cfg['train'])
    val_data = DataLoader(**data_cfg['val'])
    td, vd = train_data.prepare(), val_data.prepare()

    model_cfg = cfg['MODEL']
    class_num = len(train_data.label_dict)
    trainer = Model(exp, class_num, **model_cfg)
    train_cfg = cfg['TRAIN']
    train_cfg['train_steps'] = len(train_data) // data_cfg['train']['batch']
    train_cfg['val_steps'] = len(val_data) // data_cfg['val']['batch']
    trainer.search(td, vd, 10, train_cfg['train_steps'], train_cfg['val_steps'])
    trainer.train(td, vd, **train_cfg)


if __name__ == '__main__':
    exp = 'default'
    main(exp)
