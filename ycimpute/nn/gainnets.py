import torch

class NetD(torch.nn.Module):
    def __init__(self, feature_dim):
        """

        :param feature_dim:
        """
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, feature_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, m, g, h):
        """
        reference equation(4) in paper

        :param x: original data
        :param m: missing mask
        :param g: generated data by Generator
        :param h: hint, see paper
        :return: as a prob matrix, denote where is missing or not
        """
        self.init_weight()
        inp = m * x + (1 - m) * g
        inp = torch.cat((inp, h), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out


class NetG(torch.nn.Module):
    def __init__(self,feature_dim):
        """

        :param feature_dim:
        """
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim * 2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, feature_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]

    def forward(self, x, z, m):
        """

        reference equation(2,3) in paper

        :param x: mising data
        :param z: noise
        :param m: missing mask, used to replace missing part bu noise
        :return: generated data, size same as original data
        """
        self.init_weight()
        inp = m * x + (1 - m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out