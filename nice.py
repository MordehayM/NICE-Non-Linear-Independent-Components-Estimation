"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np



get_even = lambda xs: xs[:, 0::2]  # step=2 start=0
get_odd = lambda xs: xs[:, 1::2]  # step=2 start=1



class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions. (D)
            mid_dim: number of units in a hidden layer. (1000)
            hidden: number of hidden layers. (5)
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        # TODO fill in

        self.mask_config = mask_config
        # Divide into even and odd
        assert (mask_config in [0, 1]), "[AdditiveCoupling] mask_config type must be `0` or `1`!"
        if (mask_config == 1):
            self.first_func = get_odd  # Stay as is
            self.second_func = get_even
            self.in_dim = int(in_out_dim / 2)
            self.out_dim = int(in_out_dim / 2) + (in_out_dim % 2 > 0) #out dim for v - Coupling func
        else:
            self.first_func = get_even  # Stay as is
            self.second_func = get_odd
            self.in_dim = int(in_out_dim / 2) + (in_out_dim % 2 > 0)
            self.out_dim = int(in_out_dim / 2)


        # Coupling_func with 5 hidden layers
        Coupling_funcc = []
        Coupling_funcc.append(nn.Linear(self.in_dim, mid_dim))
        Coupling_funcc.append(nn.ReLU())
        for _ in range(hidden - 1):
            Coupling_funcc.append(nn.Linear(mid_dim, mid_dim))
            Coupling_funcc.append(nn.ReLU())
        Coupling_funcc.append(nn.Linear(mid_dim, self.out_dim))
        self.Coupling_func = nn.Sequential(*Coupling_funcc)  # unpack the list

    def forward(self, x, log_det_J=0, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # TODO fill in
        assert (reverse in [False, True]), "[AdditiveCoupling_forward] reverse type must be False or True!"

        B, W = x.size()
        x1 = self.first_func(x)
        x2 = self.second_func(x)

        y1 = x1
        if not reverse:
            y2 = x2 + self.Coupling_func(x1)
        else:
            y2 = x2 - self.Coupling_func(x1)


        # Combine odd and even
        if self.mask_config:
            x = torch.stack((y2, y1), dim=2)
        else:
            x = torch.stack((y1, y2), dim=2)
        return x.reshape((B, W)), log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimens
            ions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # TODO fill in
        self.mask_config = mask_config
        # Divide into even and odd
        assert (mask_config in [0, 1]), "[AdditiveCoupling] mask_config type must be `0` or `1`!"
        if (mask_config == 1):
            self.first_func = get_odd  # Stay as is
            self.second_func = get_even
            self.in_dim = int(in_out_dim / 2)
            self.out_dim = (int(in_out_dim / 2) + (in_out_dim % 2 > 0))*2
        else:
            self.first_func = get_even  # Stay as is
            self.second_func = get_odd
            self.in_dim = int(in_out_dim / 2) + (in_out_dim % 2 > 0)
            self.out_dim = int(in_out_dim / 2)*2

        Coupling_func = []
        Coupling_func.append(nn.Linear(self.in_dim, mid_dim))
        Coupling_func.append(nn.ReLU())
        for _ in range(hidden - 1):
            Coupling_func.append(nn.Linear(mid_dim, mid_dim))
            Coupling_func.append(nn.ReLU())
            #Coupling_func.append(nn.BatchNorm1d(mid_dim))
        Coupling_func.append(nn.Linear(mid_dim, self.out_dim))
        self.Coupling_func = nn.Sequential(*Coupling_func)  # unpack the list

        self.tanh = nn.Tanh()

    def forward(self, x, log_det_J=0, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # TODO fill in
        B, W = x.size()
        assert (reverse in [False, True]), "[AffineCoupling_forward] reverse type must be False or True!"
        x1 = self.first_func(x)
        x2 = self.second_func(x)
        y1 = x1
        b = self.Coupling_func(x1) #[:, :int(self.out_dim/2)], self.Coupling_func(x1)[:, int(self.out_dim/2):]
        #b = torch.tanh(b / 3)*10 +10 + 1e-5
        b1 = self.first_func(b)
        b1 = self.tanh(b1)
        #b1 = torch.min(b1, dim=1, keepdim=True)[0] + b1 + 1e-5
        #b1 = (b1==0).float().mul(1e-05) + b1
        #b1_e = torch.exp(b1) #prevent zero division
        #b1 = 1000*torch.sigmoid(b1) +1e-8
        #b1 = torch.tanh(b1) + 1 + 1e-5

        b2 = self.second_func(b)
        if not reverse:

            #y2 = x2*torch.exp(b1)+b2
            #y2 = x2*b1 + b2
            y2 = x2 * torch.exp(b1) + b2
            #log_det_J = log_det_J + torch.sum(b1, dim=1)
            #log_det_J = log_det_J + torch.sum(torch.log(torch.abs(b1)), dim=1)
            log_det_J = log_det_J + torch.sum(b1, dim=1)
        else:
            #y2 = (x2-b2)*torch.exp(-b1)  # if b1 not zero
            #y2 = torch.div(x2-b2, b1)
            y2 = (x2 - b2)*torch.exp(-b1)

        if self.mask_config:
            x = torch.stack((y2, y1), dim=2)

        else:
            x = torch.stack((y1, y2), dim=2)

        return x.reshape((B, W)), log_det_J


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions. (D)
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det = torch.sum(self.scale) + self.eps
        if reverse:
            h = x * torch.exp(-self.scale)
        else:
            h = x * torch.exp(self.scale)

        return h, log_det


"""Standard logistic distribution.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])
#logistic = TransformedDistribution(Uniform(torch.tensor(0, dtype=torch.float32).to(device), torch.tensor(1, dtype=torch.float32).to(device)), [SigmoidTransform().inv, AffineTransform(loc=torch.tensor(0, dtype=torch.float32).to(device), scale=torch.tensor(1, dtype=torch.float32).to(device))])
#logistic = TransformedDistribution(Uniform(torch.tensor(0, dtype=torch.float32).to(device), torch.tensor(1, dtype=torch.float32).to(device)), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])
logistic = TransformedDistribution(Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device)), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])



"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(self, prior, num_coupling, coupling_name,
                 in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_name: additive or afiine
            num_coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.num_coupling = num_coupling
        #self.coupling_name = coupling_name
        # TODO fill in
        # create an additive model
        if coupling_name == 'additive':
            self.Coupling_layers = nn.ModuleList([AdditiveCoupling(in_out_dim=in_out_dim
                                                               , mid_dim=mid_dim
                                                               , hidden=hidden
                                                               , mask_config=i % 2) for i in range(num_coupling)])

        # create an affine model
        else:
            self.Coupling_layers = nn.ModuleList([AffineCoupling(in_out_dim=in_out_dim
                                                           , mid_dim=mid_dim
                                                           , hidden=hidden
                                                           , mask_config=i % 2) for i in range(num_coupling)])
        self.scaling = Scaling(dim=in_out_dim)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        # TODO fill in

        # g(x)
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(self.num_coupling)):
            x, _ = self.Coupling_layers[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        # TODO fill in

        log_det = 0
        for i in range(len(self.Coupling_layers)):
            x, log_det = self.Coupling_layers[i](x=x, log_det_J=log_det)
        fx, log_det_scaling = self.scaling(x)
        return fx, log_det + log_det_scaling

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        #print(log_det_J.device)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1) #over D-dim
        #print(log_ll.device)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        # TODO

        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
