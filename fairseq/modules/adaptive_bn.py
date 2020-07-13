import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F

class AdaptiveBN(nn.Module):
    def __init__(self, max_nc, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        num_features = max_nc
        self.num_features = max_nc
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        training = self.training
        if training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        batch_size, nc, embed_dim = input.size()
        assert self.num_features >= nc
        output = F.batch_norm(
            input, self.running_mean[:nc], self.running_var[:nc], self.weight[:nc], self.bias[:nc],
                    self.training or not self.track_running_stats,
                    #True,
                    exponential_average_factor, self.eps)
        # we will need to use all parameters
        output += 0.0 * (torch.min(self.running_mean) + torch.min(self.running_var) + torch.min(self.weight) + torch.min(self.bias))
        return output

if __name__ == "__main__":
    torch.manual_seed(0)
    bn = AdaptiveBN(3)
    x = torch.randn(5, 3, 10)
    out = bn(x)
