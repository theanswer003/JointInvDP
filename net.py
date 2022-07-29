import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.relu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()


class Conv3dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv3d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv3d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        perturbed_outputs = F.conv3d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups) * sign_output

        self.kl = kl
        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class LinearFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Linear layer with Flipout reparameterization trick.
        Ref: https://arxiv.org/abs/1803.04386

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)

        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        # init prior mu
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # init weight and base perturbation weights
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        # sampling delta_W
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = (sigma_weight * self.eps_weight.data.normal_())

        # get kl divergence
        if return_kl:
            kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # linear outputs
        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = F.linear(x * sign_input, delta_weight,
                                     bias) * sign_output

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


#
prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0
class Net(nn.Module):
    def __init__(self, n_filters=16, nm=2):
        super(Net, self).__init__()
        # net for petrophysical parameters
        self.net1 = nn.Sequential(
            Reshape(-1, n_filters * 4, 15, 15, 28),
            nn.Conv3d(n_filters * 4, n_filters * 4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(n_filters * 4, n_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(n_filters * 2, n_filters, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(n_filters, 4, 3, padding=1),
            nn.Sigmoid()
        )

        # net for glogbal parameters
        self.fc1 = LinearFlipout(4, 4,
                                 prior_mean=prior_mu,
                                 prior_variance=prior_sigma,
                                 posterior_mu_init=posterior_mu_init,
                                 posterior_rho_init=posterior_rho_init)
        self.fc2 = LinearFlipout(4, 4,
                                 prior_mean=prior_mu,
                                 prior_variance=prior_sigma,
                                 posterior_mu_init=posterior_mu_init,
                                 posterior_rho_init=posterior_rho_init)
        self.fc3 = LinearFlipout(4, 4,
                                 prior_mean=prior_mu,
                                 prior_variance=prior_sigma,
                                 posterior_mu_init=posterior_mu_init,
                                 posterior_rho_init=posterior_rho_init)

    def forward(self, x, y):
        kl_sum = 0
        y, kl = self.fc1(y)
        kl_sum += kl
        y = torch.sigmoid(y)

        y, kl = self.fc2(y)
        kl_sum += kl
        y = torch.sigmoid(y)
        #
        y, kl = self.fc3(y)
        kl_sum += kl
        y = torch.sigmoid(y)

        return self.net1(x), y, kl_sum
