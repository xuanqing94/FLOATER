import pdb
import torch
from torch import nn

__all__ = ['flow_func']

class Conv1d_t(nn.Module):
    def __init__(self, in_c, out_c, k_size, **args):
        super().__init__()
        # We add one more channel to conv1d as time
        self.conv = nn.Conv1d(in_c + 1, out_c, k_size, **args)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, t, x):
        # x: float tensor [batch, in_c, H]
        b_x, c_x, h = x.size()
        time = torch.ones((b_x, 1, h)).to(x) * t
        x_aug = torch.cat([x, time], dim=1)
        out = self.conv(x_aug)
        return out

class Linear_t(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_f+1, out_f, bias)
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t, x):
        # x: float tensor [batch, in_c, H]
        b_x, c_x, h = x.size()
        time = torch.ones((b_x, c_x, 1)).to(x) * t
        x_aug = torch.cat([x, time], dim=2)
        out = self.linear(x_aug)
        return out

class ConvTr1d_t(nn.Module):
    def __init__(self, in_c, out_c, k_size, **args):
        super().__init__()
        # We add one more channel to conv1d as time
        self.conv_tr = nn.ConvTranspose1d(in_c + 1, out_c, k_size, **args)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.conv_tr.weight)
        nn.init.zeros_(self.conv_tr.bias)

    def forward(self, t, x):
        # x: float tensor [b, in_c, H]
        b_x, c_x, h = x.size()
        time = torch.ones((b_x, 1, h)).to(x) * t
        x_aug = torch.cat([x, time], dim=1)
        out = self.conv_tr(x_aug)
        return out

class id_flow(nn.Module):
    def __init__(self, nc, c_multiplier, embed_dim):
        super().__init__()

    def forward(self, t, x):
        return x * 0.

class flow_func(nn.Module):
    def __init__(self, nc, c_multiplier, embed_dim):
        super().__init__()
        assert embed_dim % 4 == 0
        self.embed_dim = embed_dim
        out_nc = max(int(nc * c_multiplier), 1)
        self.layers = nn.ModuleList(
            [
                Conv1d_t(nc, out_nc, 5, stride=2, padding=2),
                nn.SELU(inplace=True),
                #Conv1d_t(out_nc, out_nc, 5, stride=2, padding=2),
                #nn.SELU(inplace=True),
                #ConvTr1d_t(out_nc, out_nc, 5, stride=2, padding=2, output_padding=1),
                #nn.SELU(inplace=True),
                ConvTr1d_t(out_nc, nc, 5, stride=2, padding=2, output_padding=1),
            ]
        )

    def forward(self, t, x):
        """ x: input tensor of shape [batch, nc==n_layers, embed_dim] """
        batch, nc, w_x = x.size()
        short_cut = x
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                # It is a conv layer
                x = layer(t, x)
            else:
                # It is an activation
                x = layer(x)
        out = x + short_cut
        return out

class flow_func_linear(nn.Module):
    def __init__(self, nc, c_multiplier, embed_dim):
        super().__init__()
        assert embed_dim % 4 == 0
        self.embed_dim = embed_dim
        out_nc = max(int(nc * c_multiplier), 1)
        self.layers = nn.ModuleList(
            [
                #Conv1d_t(nc, out_nc, 5, stride=2, padding=2),
                Linear_t(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                #Conv1d_t(out_nc, out_nc, 5, stride=2, padding=2),
                #nn.SELU(inplace=True),
                #ConvTr1d_t(out_nc, out_nc, 5, stride=2, padding=2, output_padding=1),
                #nn.SELU(inplace=True),
                #ConvTr1d_t(out_nc, nc, 5, stride=2, padding=2, output_padding=1),
                Linear_t(embed_dim, embed_dim)
            ]
        )

    def forward(self, t, x):
        """ x: input tensor of shape [batch, nc==n_layers, embed_dim] """
        batch, nc, w_x = x.size()
        short_cut = x
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                # It is a conv layer
                x = layer(t, x)
            else:
                # It is an activation
                x = layer(x)
        #pdb.set_trace()
        out = x #+ short_cut
        return out

class flow_func_big(nn.Module):
    def __init__(self, nc, c_multiplier, embed_dim):
        super().__init__()
        assert embed_dim % 4 == 0
        self.embed_dim = embed_dim
        out_nc = max(int(nc * c_multiplier), 1)
        self.layers = nn.ModuleList(
            [
                Conv1d_t(nc, out_nc, 5, stride=2, padding=2),
                nn.SELU(inplace=True),
                Conv1d_t(out_nc, out_nc, 5, stride=2, padding=2),
                nn.SELU(inplace=True),
                ConvTr1d_t(out_nc, out_nc, 5, stride=2, padding=2, output_padding=1),
                nn.SELU(inplace=True),
                ConvTr1d_t(out_nc, nc, 5, stride=2, padding=2, output_padding=1),
            ]
        )

    def forward(self, t, x):
        """ x: input tensor of shape [batch, nc==n_layers, embed_dim] """
        batch, nc, w_x = x.size()
        short_cut = x
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                # It is a conv layer
                x = layer(t, x)
            else:
                # It is an activation
                x = layer(x)
        out = x + short_cut
        return out


if __name__ == "__main__":
    f = flow_func(6, 3, 512)
    x = torch.randn(3, 6, 32)
    y = f(0, x)
    assert y.size() == x.size()
    
