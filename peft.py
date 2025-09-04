import torch
import numpy as np
from functools import partial

# works for gemma
def wrap_linear(model,f):
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = f(layer.self_attn.q_proj)
        layer.self_attn.k_proj = f(layer.self_attn.k_proj)
        layer.self_attn.v_proj = f(layer.self_attn.v_proj)
        layer.self_attn.o_proj = f(layer.self_attn.o_proj)
        layer.mlp.gate_proj = f(layer.mlp.gate_proj)
        layer.mlp.up_proj = f(layer.mlp.up_proj)
        layer.mlp.down_proj = f(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = f(model.lm_head)
    return model

def get_lora_model(model,rank=8):
    class LinearWithLoRA(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            self.linear = linear
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            return self.linear(x) + 16 * x @ self.A @ self.B
    return wrap_linear(model,partial(LinearWithLoRA, rank=rank))

def get_dora_model(model,rank=8):
    class LinearWithDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + 16 * self.A @ self.B, dim=0, keepdim=True)
            return (x @ self.W + 16 * x @ self.A @ self.B) * self.mag / norm
            # W = self.W + 16 * self.A @ self.B
            # W *= self.mag / torch.linalg.norm(W, dim=0, keepdim=True)
            # return x @ W
    return wrap_linear(model,partial(LinearWithDora, rank=rank))

def get_simple_dora_model(model,rank=8):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            return (x @ self.W + 16 * x @ self.A @ self.B) * self.mag
    return wrap_linear(model,partial(LinearWithSimpleDoraTranspose, rank=rank))

def get_dora_transpose_model(model,rank=8):
    class LinearWithTransposeDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + 16 * self.A @ self.B, dim=1, keepdim=True)
            x *= (self.mag / norm).view(-1)
            return x @ self.W + 16 * x @ self.A @ self.B
    return wrap_linear(model,partial(LinearWithTransposeDora, rank=rank))

def get_simple_dora_transpose_model(model,rank=8):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            x *= self.mag.view(-1)
            return x @ self.W + 16 * x @ self.A @ self.B
    return wrap_linear(partial(LinearWithSimpleDoraTranspose, rank=rank))

def get_simple_svdora_model(model,rankU=8, rankV=8):
    class LinearWithSimpleSvdora(torch.nn.Module):
        def __init__(self, linear, rankU, rankV):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            print('here', W.shape)
            U, sigma, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
            self.U = torch.nn.Parameter(U.to(W.dtype), requires_grad=False)
            self.sigma = torch.nn.Parameter(sigma.to(W.dtype))
            self.Vh = torch.nn.Parameter(Vh.to(W.dtype), requires_grad=False)
            print(self.sigma)
            
            self.A1 = torch.nn.Parameter(torch.randn(U.shape[0], rankU, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B1 = torch.nn.Parameter(torch.zeros(rankU, U.shape[1], dtype = torch.bfloat16 ))
            self.A2 = torch.nn.Parameter(torch.randn(Vh.shape[0], rankV, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B2 = torch.nn.Parameter(torch.zeros(rankV, Vh.shape[1], dtype = torch.bfloat16 ))
        def forward(self, x):
            x = x @ self.U + 16 * x @ self.A1 @ self.B1
            x = x * self.sigma
            return x @ self.Vh + 16 * x @ self.A2 @ self.B2
    return wrap_linear(model,partial(LinearWithSimpleSvdora, rankU=rankU, rankV=rankV))
