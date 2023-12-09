from utils.ContractiveAutoEncoder import CAE
import numpy as np

cae = CAE()
losses = cae.train()
cae.save_model()
cae.load_model()
cae.extract_features()

float_array = np.array(losses)
file_path_npy = 'predictions/cae_losses.npy'
np.save(file_path_npy, float_array)