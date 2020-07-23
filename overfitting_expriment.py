import torch
def f(x):
    w = torch.Tensor([15, 10]).unsqueeze(1)
    return x.mm(w)
x_train_real = torch.Tensor([[ 0.9466, -1.9782],
        [-0.1369, -0.5264],
        [ 0.5476, -0.5469],
        [ 0.5230,  0.6096],
        [ 0.9349,  0.9042],
        [-0.2397,  0.5910],
        [ 0.5114,  0.7736],
        [ 0.9587,  1.3547],
        [ 1.0240, -0.4851],
        [ 0.1566,  0.0461]])
x_test_real = torch.Tensor([[-0.7599,  0.4363],
        [-0.1565, -0.0044],
        [-0.6557,  0.3177],
        [ 1.3036,  1.7352],
        [ 1.0575, -0.2623],
        [-0.3753, -0.3247],
        [-0.7030, -1.0614],
        [-0.8992,  3.0774],
        [-0.3803,  0.0443],
        [-1.1170, -0.0642]])

y_train = f(x_train_real)
y_test = f(x_test_real)
y_train_noise = torch.Tensor([[-0.1013],
        [ 0.0016],
        [-0.0367],
        [ 0.0830],
        [-0.0677],
        [ 0.0427],
        [-0.0780],
        [-0.0548],
        [-0.0644],
        [-0.0425]])
y_test_noise = torch.Tensor([[ 0.1607],
        [ 0.0132],
        [ 0.0391],
        [-0.0426],
        [ 0.0522],
        [ 0.0602],
        [ 0.0421],
        [-0.0411],
        [ 0.0654],
        [ 0.0045]])

y_train += y_train_noise
y_test += y_test_noise 

x_train_fake = torch.Tensor([[-7.5057e-01, -6.9199e-01,  1.1041e+00, -3.7074e-01, -4.0041e-01],
        [-9.4520e-01, -5.2456e-01, -9.9623e-01,  5.7802e-01,  1.4719e+00],
        [-5.9690e-01, -5.1541e-01, -1.3634e+00, -1.1489e+00,  1.0163e+00],
        [ 2.5889e-01, -2.0814e+00,  4.3734e-01, -6.9434e-01, -6.5733e-01],
        [-1.1121e-01,  3.5412e-01, -1.1732e-01,  9.8445e-01, -1.6411e-01],
        [-6.0028e-01,  1.9403e+00, -9.9754e-01,  9.8376e-01, -7.1853e-01],
        [ 1.2613e+00,  4.0883e-01, -8.9433e-01,  2.1045e+00,  1.9318e+00],
        [ 8.1148e-02, -4.4586e-01,  2.6531e+00,  3.8037e-01,  3.9278e-01],
        [ 2.2733e+00, -2.2315e+00, -3.5281e-01, -9.4487e-01,  6.0345e-02],
        [ 1.8862e-01, -3.7595e-01, -5.0980e-01,  1.7804e+00, -6.1080e-03]])
x_test_fake = torch.Tensor([[ 2.8540e-01,  4.2668e-01, -6.8618e-01,  9.8210e-01, -1.5898e+00],
        [ 2.9958e-01,  6.7035e-01,  2.1784e-01, -8.5531e-01, -7.4837e-01],
        [-1.1771e+00,  1.2218e+00,  2.9322e-01, -2.1670e+00, -6.4644e-01],
        [-9.9442e-01,  1.4766e-01,  1.0723e+00,  1.6482e+00, -2.0126e+00],
        [-1.1844e+00,  2.1204e-01, -3.3065e-01,  3.2540e-01,  8.3172e-01],
        [ 1.2190e-01,  4.7056e-02,  1.6156e+00, -1.1533e+00, -1.7905e-01],
        [-1.5765e+00, -9.8900e-01, -1.3708e-01,  1.5952e-01, -3.3339e-01],
        [ 3.9933e-01,  1.5456e+00,  4.5935e-01,  1.8694e-04, -4.3587e-01],
        [ 2.8316e-01, -2.9840e-01, -1.9063e+00,  5.8378e-01,  9.4472e-01],
        [ 1.8540e+00,  4.2780e-01, -2.9085e-01, -7.2814e-01, -7.5511e-01]])

def make_feature(x_real, x_fake=None, use_x_fake=False, use_x2=False):
    x1 = x_real[:, :1]
    x2 = x_real[:, 1:2]
    result = x1
    if use_x2:
        result = torch.cat([result, x2], 1)
    if use_x_fake:
        result = torch.cat([result, x_fake, x1 ** 3, x1 ** 5, x1 ** 7, x1 ** 9], 1)
    return result
    
from torch import nn
class poly_model(nn.Module):
    def __init__(self, use_x_fake=False, use_x2=False):
        super(poly_model,self).__init__()
        n_input_features = 1
        if use_x2:
            n_input_features += 1
        if use_x_fake:
            n_input_features += 9
        self.poly = nn.Linear(n_input_features ,1)
    def forward(self,input):
        output = self.poly(input)
        return output

import matplotlib.pyplot as plt
def train_model(use_x_fake=False, use_x2=False):
    model = poly_model(use_x_fake, use_x2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    N_EPOCH = 100000
    losses_train = []
    losses_test = []
    for epoch in range(N_EPOCH):
        x_train = make_feature(x_train_real, x_fake=x_train_fake, use_x_fake=use_x_fake, use_x2=use_x2)
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    
        x_test = make_feature(x_test_real, x_fake=x_train_fake, use_x_fake=use_x_fake, use_x2=use_x2)
        with torch.no_grad():
            output_test = model(x_test)
            loss_test = criterion(output_test, y_test)
        
        if epoch % 1000 == 0:
            name, param = list(model.named_parameters())[0]
            losses_train.append(loss_train.data)
            losses_test.append(loss_test.data)
            print("epoch = %d, loss = %f, loss_test = %f, weights =" %(epoch, loss_train.data, loss_test.data), param.data)
    plt.plot(losses_train, '-b', label="loss_train")
    plt.plot(losses_test, '-r', label="loss_test")
    plt.legend(loc='upper right')
    plt.show()
    
    
train_model(use_x2=False, use_x_fake=False)
train_model(use_x2=True, use_x_fake=False)
train_model(use_x2=False, use_x_fake=True)
train_model(use_x2=True, use_x_fake=True)
