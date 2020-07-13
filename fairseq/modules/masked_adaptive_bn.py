import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
import pdb


class MaskedAdaptiveN(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1.0e-5):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long)) 
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
    
    def reset_parameters(self):
        self.reset_running_stats()
        init.ones_(self.weight)
        init.zeros_(self.bias)
    
    def forward(self, input, *unused, **unusedk):
        bsz, nc, embed_dim = input.size()
        output = F.batch_norm(
            input, self.running_mean[:nc], self.running_var[:nc], self.weight[:nc], self.bias[:nc], self.training, 0.1, 1.0e-5)
        # we will need to use all parameters
        return output

class MaskedAdaptiveBN(nn.Module):
    def __init__(self, max_nc, momentum=0.1, eps=1.0e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.max_nc = max_nc
        self.gamma = Parameter(torch.Tensor(1, max_nc, 1))
        self.beta = Parameter(torch.Tensor(1, max_nc, 1))
        self.register_buffer("running_mean", torch.Tensor(1, max_nc, 1))
        self.register_buffer("running_std", torch.Tensor(1, max_nc, 1))
        self.register_buffer("r_max", torch.tensor(1.00))
        self.register_buffer("d_max", torch.tensor(0.00))

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        init.zeros_(self.running_mean)
        init.ones_(self.running_std)

    def forward(self, x, mask):
        """
        x: tensor of shape [bsz, seq_len, embed_dim]
        mask: 0/1 tensor of shape [bsz, seq_len]
        When calculating running stats, we ignore the masked position
        """
        bsz, seq_len, embed_dim = x.size()
        running_mean = self.running_mean[:, :seq_len, :]
        running_std = self.running_std[:, :seq_len, :]
        if self.training:
            if mask is not None:
                mask = mask.unsqueeze(dim=-1)
                inv_mask = 1.0 - mask.to(x)
            else:
                inv_mask = torch.ones(bsz, seq_len, 1).to(x)
            # Count non-pad
            nonpads = torch.sum(inv_mask, dim=0, keepdim=True) * embed_dim
            batch_mean = torch.sum(x * inv_mask, dim=(0, 2), keepdim=True) / nonpads
            batch_var = torch.sum(
                x.pow(2) * inv_mask, dim=(0, 2), keepdim=True
            ) / nonpads - batch_mean.pow(2)
            batch_std = torch.sqrt(torch.clamp(batch_var, self.eps, 1e10))
            r = torch.clamp(
                batch_std / running_std, 1.0 / self.r_max, self.r_max
            ).detach()
            d = torch.clamp((batch_mean - running_mean) / running_std, -self.d_max, self.d_max).detach()
            
            x_normalized = (x - batch_mean) * r / batch_std + d
            #x_normalized = (x - batch_mean) / batch_std
            x_normalized = (
                x_normalized * self.gamma[:, :seq_len, :] + self.beta[:, :seq_len, :]
            )

            # Update running mean and std
            self.running_mean[:, :seq_len, :] = (
                1.0 - self.momentum
            ) * running_mean + self.momentum * batch_mean.data
            self.running_std[:, :seq_len, :] = (
                1.0 - self.momentum
            ) * running_std + self.momentum * batch_std.data
        else:
            x_normalized = (x - running_mean) / running_std
            x_normalized = (
                x_normalized * self.gamma[:, :seq_len, :] + self.beta[:, :seq_len, :]
            )
        return x_normalized


if __name__ == "__main__":
    bn = MaskedAdaptiveBN(177)
    input = torch.randn(15, 13, 128, requires_grad=True)
    mask = torch.ones(15, 13)
    out = bn(input, mask)
    assert out.size() == input.size()
    loss = torch.mean(out * out)
    loss.backward()
