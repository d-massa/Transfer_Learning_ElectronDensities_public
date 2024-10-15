import torch
from transformers import CLIPModel
import argparse
import yaml
from dataloader import dataprep
from train import experiment,kfold_train, holdout_performance, kfold_train_val_monit
from transformers import CLIPProcessor

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)

    paths_local=[config['images_type'],
                '../data/data_cleaning/Newdata_normalized_zscore_interval/filtered_FCC_properties.csv',
                '../data/data_cleaning/Newdata_normalized_zscore_interval/filtered_FCC_descriptions.csv']

    #paths_cluster=['../data/valid_data_plots','../data/FCC_properties_w_dens_valid.csv','../data/FCC_descriptions_valid.csv']

    paths=paths_local

    clip_model = CLIPModel.from_pretrained(config['clip_model'])
    clip_processor = CLIPProcessor.from_pretrained(config['clip_model'])
    
    #try smaller patch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    train_val_dataset, holdout_dataset = dataprep(paths,config['holdout_ratio'],config['seedd'],config['target'],config['augmentation'],clip_processor,config['images_prefix'],config['normalization'],nrows=0)
    print('LENGTHS:', len(train_val_dataset),len(holdout_dataset))

    experiment(config['out_suffix'])
    mean_r2,mean_rmse=kfold_train_val_monit(config['epochs'],config['k_folds'],config['seedd'],train_val_dataset,config['batch_size'],clip_model,config['learning_rate'],device,config['out_suffix'],config['patience'])
    holdout_r2,holdout_rmse=holdout_performance(holdout_dataset,config['batch_size'],clip_model,device,config['out_suffix'])

    print(f'Final results:\n Mean R^2 and RMSE across folds: {mean_r2:.4f}  {mean_rmse:.4f},\n Holdout R^2 and RMSE: {holdout_r2:.4f}  {holdout_rmse:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with a specific configuration file")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args.config)