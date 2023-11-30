import pandas as pd
import numpy as np
import argparse
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 5,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 961,
              'num_items': 1000,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 1,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 961,
              'num_items': 1000,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 7,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format(
                'gmf_factor8neg4-implict_Epoch5_HR0.0844_NDCG0.0414.model'), # gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 500,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 961,
                'num_items': 1000,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 7,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format(
                  'gmf_factor8neg4-implict_Epoch5_HR0.0844_NDCG0.0414.model'),
                'pretrain_mlp': 'checkpoints/{}'.format(
                  'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_Epoch1_HR0.0966_NDCG0.0485.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

def parse_args():
  parser = argparse.ArgumentParser(description="Training")
  parser.add_argument('--data_dir', type=str, required=True, 
                      help="Path to the MovieLens 1M ratings.dat file")
  parser.add_argument('--model', type=str, required=True, choices=['gmf', 'mlp', 'neumf'],
                      help="Choose the recommendation model to run: 'gmf', 'mlp', or 'neumf'")
  return parser.parse_args()

def main():
  args = parse_args()

  # Load Data
  print('Loading Data....')
  user_item_interactions = df = pd.read_json(args.data_dir, lines=True)
  df = pd.DataFrame(user_item_interactions)
  df = df.groupby(['user_id', 'business_id']).agg({'stars': 'mean'}).reset_index()

  user_id = df[['user_id']].drop_duplicates().reindex()
  user_id['userId'] = np.arange(len(user_id))
  ml1m_rating = pd.merge(df, user_id, on=['user_id'], how='left')

  item_id = df[['business_id']].drop_duplicates()
  item_id['itemId'] = np.arange(len(item_id))
  yelp_rating = pd.merge(ml1m_rating, item_id, on=['business_id'], how='left')
  yelp_rating = yelp_rating[['userId', 'itemId', 'stars']]
  yelp_rating.rename(columns={'stars': 'rating'}, inplace=True)
  # print(yelp_rating.head())

  # DataLoader for training
  sample_generator = SampleGenerator(ratings=yelp_rating)
  evaluate_data = sample_generator.evaluate_data
  print(evaluate_data)

  # Specify the exact model based on the command line argument
  if args.model == 'gmf':
    config = gmf_config
    engine = GMFEngine(config)
  elif args.model == 'mlp':
    config = mlp_config
    engine = MLPEngine(config)
  elif args.model == 'neumf':
    config = neumf_config
    engine = NeuMFEngine(config)

  for epoch in range(config['num_epoch']+1):
      print('Epoch {} starts !'.format(epoch))
      print('-' * 80)
      train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
      engine.train_an_epoch(train_loader, epoch_id=epoch)

      hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
      
      # engine.save(config['alias'], epoch, hit_ratio, ndcg)
      # print('Model saved at epoch {}'.format(epoch))      
      
      if epoch % 50 == 0:
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        print('Model saved at epoch {}'.format(epoch))

if __name__ == "__main__":
    main()
