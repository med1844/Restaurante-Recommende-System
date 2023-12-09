from utils.ContractiveAutoEncoder import CAE

cae = CAE()
cae.train()
cae.save_model()
cae.load_model()
cae.extract_features()