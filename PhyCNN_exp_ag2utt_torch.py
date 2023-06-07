import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os

class CNN(nn.Module):
    def __init__(self, input_size, feature_size, numout):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=1, padding=25),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=50, stride=1, padding=25),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=50, stride=1, padding=25),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=50, stride=1, padding=25),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=50, stride=1, padding=25),
            nn.ReLU()
        )

        # n_channels = self.conv(torch.empty(1, input_size, feature_size)).size(-1)
        self.output_layer = nn.Sequential(nn.Linear(64, 50),
                                          nn.ReLU(),
                                          nn.Linear(50, 50),
                                          nn.ReLU(),
                                          nn.Linear(50, numout))

    def forward(self, x_in):
        x = x_in.permute(0, 2, 1)
        x = self.conv(x)
        x = x[:, :, :2500]
        x = x.permute(0, 2, 1)
        out = self.output_layer(x)# (15, 2500, 2) 출력 형태로 변환
        return out


class DeepPhyLSTM:
    # Initialize the class
    def __init__(self, eta_tt, ag, Phi_t):
        super(DeepPhyLSTM, self).__init__()

        self.eta_tt = eta_tt

        self.ag = ag
        self.Phi_t = Phi_t

        # placeholders for data
        self.learning_rate = 0.001 # 초기 학습률 지정
        self.eta_tt_torch = torch.from_numpy(eta_tt).float()
        self.ag_torch = torch.from_numpy(ag).float()
        
        self.model = CNN(input_size=1, feature_size=ag.shape[-1], numout=self.eta_tt.shape[2])
        self.eta_tt_torch.requires_grad = True
        self.ag_torch.requires_grad = True
        

        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def net_structure(self, ag):

        eta = self.model(ag)
        Phi_ut = self.Phi_t.reshape(1, self.eta_tt.shape[1], self.eta_tt.shape[1])
        Phi_ut = Phi_ut.repeat(self.eta_tt.shape[0], axis=0)
        
        eta_t = torch.matmul(torch.tensor(Phi_ut, dtype=torch.float32), eta)
        eta_tt = torch.matmul(torch.tensor(Phi_ut, dtype=torch.float32), eta_t)

        return eta, eta_t, eta_tt

    def criterion(self, eta_tt_torch, eta_tt_pred, eta_pred):
        return torch.mean(torch.square(eta_tt_torch - eta_tt_pred)) + torch.mean(torch.square(eta_pred[:,:,0:10]))

    def train(self, num_epochs, learning_rate, bfgs, batch_size = 64):
        Loss = []
        self.eta_pred, self.eta_t_pred, self.eta_tt_pred = self.net_structure(self.ag_torch)
        
        self.learning_rate = learning_rate
        self.model.train()

        for epoch in range(num_epochs):
            N = self.eta_tt.shape[0]
            for it in range(0, N, batch_size):
                self.optimizer_Adam.zero_grad()
                self.eta_pred, self.eta_t_pred, self.eta_tt_pred = self.net_structure(self.ag_torch)
                loss = self.criterion(self.eta_tt_torch, self.eta_tt_pred, self.eta_pred)
                loss.backward()
                self.optimizer_Adam.step()
                if it % (10*batch_size) == 0:
                        print('Epoch: %d, It: %d, Loss: %.3e,  Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss.item(), learning_rate))

            Loss.append(loss.item())

        if bfgs == 1:
            # use L-BFGS optimizer
            self.optimizer_lbfgs = optim.LBFGS(self.model.parameters(), lr=self.learning_rate, max_iter=2000, max_eval=50000, history_size=50, 
                                               line_search_fn='strong_wolfe')
            def closure():
                self.optimizer_lbfgs.zero_grad()
                self.eta_pred, self.eta_t_pred, self.eta_tt_pred = self.net_structure(self.ag_torch)
                loss = self.criterion(self.eta_tt_torch, self.eta_tt_pred, self.eta_pred)
                loss.backward()
                return loss

            self.optimizer_lbfgs.step(closure)

            print('LBFGS Loss :', self.criterion(self.eta_tt_torch, self.eta_tt_pred, self.eta_pred).item())
            Loss.append(self.criterion(self.eta_tt_torch, self.eta_tt_pred, self.eta_pred).item())
            

        return Loss

    def predict(self, ag_test):
        ag_test_torch = torch.from_numpy(ag_test).float()
        ag_test_torch.requires_grad = True

        eta_pred, eta_t_pred, eta_tt_pred = self.net_structure(ag_test_torch)

        eta_pred = eta_pred.detach().numpy()
        eta_t_pred = eta_t_pred.detach().numpy()
        eta_tt_pred = eta_tt_pred.detach().numpy()

    # def predict(self, ag_star):
    #     ag_star = torch.tensor(ag_star, dtype=torch.float32)
    #     eta_pred, eta_t_pred, eta_tt_pred = self.net_structure(ag_star)
    #     return eta_pred.detach().numpy(), eta_t_pred.detach().numpy(), eta_tt_pred.detach().numpy()

        return eta_pred, eta_t_pred, eta_tt_pred




if __name__ == "__main__":

    # load data
    dataDir = "D:/MHC/2022/개인자료/kmong/battery_health_등푸른비행선/tf to torch/PhyCNN/data/"
    mat = scipy.io.loadmat(dataDir + 'data_exp.mat')

    ag_data = mat['input_tf'][:, 0:2500]
    u_data = mat['target_X_tf'][:, 0:2500, :]
    ut_data = mat['target_Xd_tf'][:, 0:2500, :]
    utt_data = mat['target_Xdd_tf'][:, 0:2500, :]
    train_indices = mat['trainInd'] - 1
    test_indices = mat['valInd'] - 1

    ag_data = np.reshape(ag_data, [ag_data.shape[0], ag_data.shape[1], 1])

    ag_train = ag_data
    eta_train = u_data
    eta_t_train = ut_data
    eta_tt_train = utt_data

    dt = 0.02

    ag_all = ag_data
    u_all = u_data
    u_t_all = ut_data
    u_tt_all = utt_data

    # finite difference
    n = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
    Phi_t = 1 / dt * np.concatenate(
            [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)

    ag_star = ag_all
    eta_star = u_all
    eta_t_star = u_t_all
    eta_tt_star = u_tt_all
    g_star = -eta_tt_star -ag_star
    lift_star = -ag_star

    N_train = eta_star.shape[0]

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    g = g_star

    # Training Data
    eta_train = eta
    ag_train = ag
    lift_train = lift
    eta_t_train = eta_t
    eta_tt_train = eta_tt
    g_train = g

    model = DeepPhyLSTM(eta_tt_train, ag_train, Phi_t)

    Loss = model.train(num_epochs=100, learning_rate=1e-3, bfgs=1, batch_size=N_train)
    train_loss = Loss

    plt.figure()
    plt.plot(np.log(train_loss), label='loss')
    plt.legend()
    # Training performance

    X_train = ag_train
    y_train_ref = eta_train
    yt_train_ref = eta_t_train
    ytt_train_ref = eta_tt_train
    # g_train_ref = -eta_tt_train-ag_train

    # Prediction
    eta, eta_t, eta_tt = model.predict(X_train)

    y_train_pred = eta
    yt_train_pred = eta_t
    ytt_train_pred = eta_tt

    dof = 0
    for n in range(len(ag_star)):
        plt.figure()
        plt.plot(y_train_ref[n, :, dof], label='True')
        plt.plot(y_train_pred[n, :, dof], label='Predict')
        plt.title('Training_u')
        plt.legend()

    for n in range(len(ag_star)):
        plt.figure()
        plt.plot(yt_train_ref[n, :, dof], label='True')
        plt.plot(yt_train_pred[n, :, dof], label='Predict')
        plt.title('Training_ut')
        plt.legend()

    for n in range(len(ag_star)):
        plt.figure()
        plt.plot(ytt_train_ref[n, :, dof], label='True')
        plt.plot(ytt_train_pred[n, :, dof], label='Predict')
        plt.title('Training_utt')
        plt.legend()

    # Prediction performance
    ag_pred = mat['input_pred_tf']
    u_pred = mat['target_pred_X_tf']
    ut_pred = mat['target_pred_Xd_tf']
    utt_pred = mat['target_pred_Xdd_tf']
    ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])

    X_pred = np.concatenate([ag_pred, ag_pred, ag_pred[0:3]], axis=0)[:, 0:2500, :]
    y_pred_ref = np.concatenate([u_pred, u_pred, u_pred[0:3]], axis=0)[:, 0:2500, :]
    yt_pred_ref = np.concatenate([ut_pred, ut_pred, ut_pred[0:3]], axis=0)[:, 0:2500, :]
    ytt_pred_ref = np.concatenate([utt_pred, utt_pred, utt_pred[0:3]], axis=0)[:, 0:2500, :]

    # Prediction
    eta, eta_t, eta_tt = model.predict(X_pred)
    y_pred = eta
    yt_pred = eta_t
    ytt_pred = eta_tt

    for ii in range(len(y_pred)):
        plt.figure()
        plt.plot(y_pred_ref[ii, :, dof], label='True')
        plt.plot(y_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_u')
        plt.legend()

        plt.figure()
        plt.plot(yt_pred_ref[ii], label='True')
        plt.plot(yt_pred[ii], label='Predict')
        plt.title('Prediction_u_t')
        plt.legend()

        plt.figure()
        plt.plot(ytt_pred_ref[ii, :, dof], label='True')
        plt.plot(ytt_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_u_tt')
        plt.legend()

    R1 = []
    R2 = []
    R3 = []
    for ii in range(len(y_pred_ref)):
        reg1 = LinearRegression().fit(y_pred_ref[ii, :, 0:1], y_pred[ii, :, 0:1])
        R1.append(reg1.coef_)
    R = np.concatenate([np.array(R1)])
    R = R[:, 0]
    plt.hist(R, 'auto', density=True, facecolor='blue', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram_Model')
    ax = plt.gca()
    ax.invert_xaxis()

scipy.io.savemat(dataDir + 'results/results_exp_ag2utt.mat',
                 {'y_train_ref': y_train_ref, 'yt_train_ref': yt_train_ref, 'ytt_train_ref': ytt_train_ref,
                  'y_train_pred': y_train_pred, 'yt_train_pred': yt_train_pred, 'ytt_train_pred': ytt_train_pred,
                  'y_pred_ref': y_pred_ref, 'yt_pred_ref': yt_pred_ref, 'ytt_pred_ref': ytt_pred_ref,
                  'y_pred': y_pred, 'yt_pred': yt_pred, 'ytt_pred': ytt_pred,
                  'X_train': X_train, 'X_pred': X_pred, 'dt': dt,
                  'train_loss': train_loss})