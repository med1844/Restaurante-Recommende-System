import numpy as np
import pandas as pd
import os

class AutoSVDpp():
    '''
    This is an efficient version of AutoSVD++ which uses sparse matrix.
    '''
    def __init__(self, path_of_feature_file, epochs=30, num_factors=10, 
                 gamma1=0.007, gamma2 = 0.007, lambda1=0.005, lambda2=0.015, beta=0.1):
        self.epochs = epochs
        self.num_factors = num_factors
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.lambda1 =lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.path_of_feature_file = path_of_feature_file

    def train(self, train_data, test_data):
        users, items = train_data.nonzero()
        uni_users = np.unique(users)

        num_of_ratings = len(users) #len(items)
        sum_of_ratings = 0
        item_by_users = {}
        for u, i in zip(users,items):
            sum_of_ratings += train_data[u,i]
            item_by_users.setdefault(u,[]).append(i)
        average_rating_train = float(sum_of_ratings/num_of_ratings)
        self.average_rating = average_rating_train
        self.item_by_users = item_by_users

        m, n = train_data.shape
        self.item_features = self.readItemFeature()

        np.random.seed(6220)
        
        V = np.random.rand(self.num_factors, n) * 0.01
        U = np.random.rand(self.num_factors, m) * 0.01
        Y = np.random.rand(n,self.num_factors) * 0.0
        B_U = np.zeros(m)
        B_I = np.zeros(n)

        result = 0
        self.rmse_list = []
        self.mae_list = []

        for epoch in range(self.epochs):
            print("[epoch=" + str(epoch + 1) + "] ", end="")
            count = 0
            for u in uni_users:
                n_u = len(item_by_users[u])
                sqrt_n_u = np.sqrt(n_u)
                sum_y_j = np.zeros(self.num_factors)

                for j in item_by_users[u]:
                    sum_y_j += Y[j, :]
                p_im = sum_y_j / sqrt_n_u
                p_old = p_im

                for i in item_by_users[u]:
                    dot = np.dot((p_im + U[:, u]).T, (V[:, i] + self.beta * self.item_features[i]))
                    error = train_data[u,i] - (average_rating_train + B_U[u] + B_I[i] + dot)
                    #update parameters
                    B_U[u] += self.gamma1 * (error - self.lambda1 * B_U[u])
                    B_I[i] += self.gamma1 * (error - self.lambda1 * B_I[i])
                    V[:, i] += self.gamma2 * (error * (p_im + U[:,u]) - self.lambda2 * V[:,i])
                    U[:, u] += self.gamma2 * (error * (V[:,i] + self.beta * self.item_features[i]) - self.lambda2 * U[:,u])
                    p_im += self.gamma2 * (error  * (V[:,i] + self.beta * self.item_features[i]) -  self.lambda1 * p_im)
                for item in item_by_users[u]:
                    Y[item, :] +=   (1/sqrt_n_u)  * ( p_im - p_old)
            
            self.B_U = B_U
            self.B_I = B_I
            self.V = V
            self.U = U
            self.Y = Y
            rmse, mae = self.evaluate(test_data)
            self.rmse_list.append(rmse)
            self.mae_list.append(mae)

        # self.B_U = B_U
        # self.B_I = B_I
        # self.V = V
        # self.U = U
        # self.Y = Y

    def evaluate(self, data_set):
        users, items = data_set.nonzero()
        num_of_ratings = len(users)
        sum_for_rmse = 0
        sum_for_mae = 0
        for u, i in zip(users, items):
            if u in self.item_by_users:
                n_u = len(self.item_by_users[u])
                sqrt_n_u = np.sqrt(n_u)
                sum_y_j = np.zeros(self.num_factors)
                for j in self.item_by_users[u]:
                    sum_y_j += self.Y[j, :]
                dot = np.dot((sum_y_j / sqrt_n_u + self.U[:, u]).T, (self.V[:, i] + self.beta * self.item_features[i]))
            else:
                dot = np.dot(( self.U[:, u]).T, (self.V[:, i] + self.beta * self.item_features[i]))
            error = data_set[u,i] - (self.average_rating + self.B_U[u] + self.B_I[i] + dot)
            sum_for_rmse += error ** 2
            sum_for_mae += abs(error)

        rmse = np.sqrt(sum_for_rmse /num_of_ratings)
        mae = sum_for_mae/num_of_ratings

        print("AutoSVD++ RMSE = {:.5f},".format(rmse), "AutoSVD++ MAE = {:.5f}".format(mae))

        return rmse,mae
  
    def save_model(self, directory='parameters/'):
        # Save the matrix to the file
        np.savetxt(os.path.join(directory, "B_U_pp.csv"), self.B_U, delimiter=',')
        np.savetxt(os.path.join(directory, "B_I_pp.csv"), self.B_I, delimiter=',')
        np.save(os.path.join(directory, "V_pp.npy"), self.V)
        np.save(os.path.join(directory, "U_pp.npy"), self.U)
        np.save(os.path.join(directory, "Y_pp.npy"), self.Y)
        np.save(os.path.join(directory, "beta_pp.npy"), self.beta)
        np.save(os.path.join(directory, "average_rating_pp.npy"), self.average_rating)
        # Save rmse and mae during training
        rmse_array = np.array(self.rmse_list, dtype=float)
        np.savetxt(os.path.join(directory, "rmse_pp.csv"), rmse_array, delimiter=',', fmt='%s')
        mae_array = np.array(self.mae_list, dtype=float)
        np.savetxt(os.path.join(directory, "mae_pp.csv"), mae_array, delimiter=',', fmt='%s')

    def load_model(self, directory='parameters/'):
        # load the matrix from the file
        self.B_U = np.loadtxt(os.path.join(directory, "B_U_pp.csv"))
        self.B_I = np.loadtxt(os.path.join(directory, "B_I_pp.csv"))
        self.V = np.load(os.path.join(directory, "V_pp.npy"))
        self.U = np.load(os.path.join(directory, "U_pp.npy"))
        self.Y = np.load(os.path.join(directory, "Y_pp.npy"))
        self.beta = np.load(os.path.join(directory, "beta_pp.npy"))
        self.average_rating = np.load(os.path.join(directory, "average_rating_pp.npy"))

    def readItemFeature(self):
        df_restaurants = pd.read_csv(self.path_of_feature_file)
        df_restaurants = df_restaurants.drop(columns=['Unnamed: 0'])
        item_features = [np.array(row, dtype=np.float64) for _, row in df_restaurants.iterrows()]
        return item_features
