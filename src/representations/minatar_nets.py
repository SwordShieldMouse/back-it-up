import torch 
import torch.nn as nn 

class MinAtarVQPolicy(nn.Module):
    """ combined action value, value, and policy """ 
    def __init__(self, action_dim, n_hidden, in_channels, combo=["v", "q", "pi"], separate_bodies = True):
        super(MinAtarVQPolicy, self).__init__()
        self.separate_bodies = separate_bodies
        #print(in_channels)
        
        if self.separate_bodies is False:
            self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

            # code from MinAtar paper
            # Final fully connected hidden layer:
            #   the number of linear unit depends on the output of the conv
            #   the output consist 128 rectified units
            def size_linear_unit(size, kernel_size=3, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1
            num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

            if "v" in combo:
                self.v = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(num_linear_units, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, 1)
                )
            if "q" in combo:
                self.q = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(num_linear_units, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, action_dim)
                )

            if "pi" in combo:
                self.policy_net = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(num_linear_units, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, action_dim),
                    nn.Softmax(dim = -1)
                )
            #self.network.apply(utils.init_weights)
        else:
            self.q = MinAtarQ(action_dim, n_hidden, in_channels)
            self.v = MinAtarV(n_hidden, in_channels)
            self.policy_net = MinAtarPolicy(action_dim, n_hidden, in_channels)
        self.combo = combo

    def forward(self, x):
        #print(x.shape)
        if self.separate_bodies is False:
            if len(x.shape) == 3:
                x = self.conv(x.permute(2, 0, 1).unsqueeze(0)).view(-1)
            elif len(x.shape) == 4:
                x = self.conv(x.permute(0, 3, 1, 2)).view((x.shape[0], -1))
        res = []
        if "v" in self.combo:
            res.append(self.v(x))
        if "q" in self.combo:
            res.append(self.q(x))
        if "pi" in self.combo:
            res.append(self.policy_net(x))
        if len(res) == 1:
            return res[0]
        else:
            return res


class MinAtarV(nn.Module):
    def __init__(self, n_hidden, in_channels):
        super(MinAtarV, self).__init__()

        #print(in_channels)
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # code from MinAtar paper
        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_linear_units, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

        #self.network.apply(utils.init_weights)

    def forward(self, x):
        #print(x.shape)
        if len(x.shape) == 3:
            x = self.conv(x.permute(2, 0, 1).unsqueeze(0)).view(-1)
        elif len(x.shape) == 4:
            x = self.conv(x.permute(0, 3, 1, 2)).view((x.shape[0], -1))
        return self.net(x)

class MinAtarQ(nn.Module):
    def __init__(self, action_dim, n_hidden, in_channels):
        super(MinAtarQ, self).__init__()

        #print(in_channels)
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # code from MinAtar paper
        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_linear_units, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_dim)
        )

        #self.network.apply(utils.init_weights)

    def forward(self, x):
        if len(x.shape) == 3:
            x = self.conv(x.permute(2, 0, 1).unsqueeze(0)).view(-1)
        elif len(x.shape) == 4:
            # print(x.shape)
            x = self.conv(x.permute(0, 3, 1, 2)).view((x.shape[0], -1))
            # print(x.shape)
        return self.net(x)

class MinAtarPolicy(nn.Module):
    def __init__(self, action_dim, n_hidden, in_channels):
        super(MinAtarPolicy, self).__init__()

        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # code from MinAtar paper
        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_linear_units, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_dim),
            nn.Softmax(dim = -1)
        )

        #self.net.apply(utils.init_weights)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 3: 
            # not batch
            # print(x.shape)
            x = self.conv(x.permute(2, 0, 1).unsqueeze(0)).view(-1)
            # print(x.shape)
        elif len(x.shape) == 4:
            # batch
            x = self.conv(x.permute(0, 3, 1, 2)).view((x.shape[0], -1))
        return self.net(x)