import torch
import numpy as np
import functools

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

def wrap_like_linear(model,f_factory):
    for param in model.parameters():
        param.requires_grad = False
    f = f_factory(model.model.layers[0].self_attn.q_proj.in_features,
                                model.model.layers[0].self_attn.q_proj.out_features)
    for layer in model.model.layers:
        layer.self_attn.q_proj = f(layer.self_attn.q_proj)
    f = f_factory(model.model.layers[0].self_attn.k_proj.in_features,
                                model.model.layers[0].self_attn.k_proj.out_features)
    for layer in model.model.layers:
        layer.self_attn.k_proj = f(layer.self_attn.k_proj)
    f = f_factory(model.model.layers[0].self_attn.v_proj.in_features,
                                model.model.layers[0].self_attn.v_proj.out_features)
    for layer in model.model.layers:
        layer.self_attn.v_proj = f(layer.self_attn.v_proj)
    f = f_factory(model.model.layers[0].self_attn.o_proj.in_features,
                                model.model.layers[0].self_attn.o_proj.out_features)
    for layer in model.model.layers:
        layer.self_attn.o_proj = f(layer.self_attn.o_proj)
    f = f_factory(model.model.layers[0].mlp.gate_proj.in_features,
                                model.model.layers[0].mlp.gate_proj.out_features)
    for layer in model.model.layers:
        layer.mlp.gate_proj = f(layer.mlp.gate_proj)
    f = f_factory(model.model.layers[0].mlp.up_proj.in_features,
                                model.model.layers[0].mlp.up_proj.out_features)
    for layer in model.model.layers:
        layer.mlp.up_proj = f(layer.mlp.up_proj)
    f = f_factory(model.model.layers[0].mlp.down_proj.in_features,
                                model.model.layers[0].mlp.down_proj.out_features)
    for layer in model.model.layers:
        layer.mlp.down_proj = f(layer.mlp.down_proj)
    return model

# considers a set of n linear parameters as a three place tensor of shape 
# (in_features, out_features, n). We implicitly embed a tensor of shape (a,b,l)
# here defining our fine tune (consider a << in_features, b << out_features, l
# << n). For the special case l = n, a simpler implementation without loss of
# expressivity is avaiable in get_tied_lora_extra_wrapper
def create_tensor_embedding_wrapper(in_features, out_features, a, b, l, premult=True, postmult=True, dtype=torch.float32):
    T = torch.nn.Parameter(torch.zeros(a, b, l, dtype=dtype))
    A = torch.nn.Parameter(torch.randn(a, in_features, dtype=dtype))
    B = torch.nn.Parameter(torch.randn(b, out_features, dtype=dtype))
    class LinearWithTensorEmbedding(torch.nn.Module):
        def __init__(self, linear, T, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.T = T
            self.A = A
            self.B = B
            self.M = torch.nn.Parameter(torch.randn(l, dtype=dtype))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            if premult:
                x = x * self.pre
            y = torch.einsum('...i,ai,bo,l,abl->...o', x, self.A, self.B, self.M, self.T)
            if postmult:
                y = y * self.post
            return self.linear(x) + y
    return functools.partial(LinearWithTensorEmbedding, T=T, A=A, B=B)

def get_tensor_contraction_model(model,a=8,b=8,l=8,premult=False,postmult=False):
    return wrap_like_linear(model, functools.partial(create_tensor_embedding_wrapper, a=a, b=b, l=l, 
                                                     premult=premult, postmult=postmult, dtype=model.dtype))
    
def create_tied_lora_extra_wrapper(in_features, out_features, a, b, premult=True, postmult=True, dtype=torch.float32):
    A = torch.nn.Parameter(torch.randn(a, in_features, dtype=dtype))
    B = torch.nn.Parameter(torch.randn(b, out_features, dtype=dtype))
    class LinearWithTiedLoraExtra(torch.nn.Module):
        def __init__(self, linear, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.A = A
            self.B = B
            self.M = torch.nn.Parameter(torch.zeros(a,b, dtype=dtype))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            if premult:
                x = x * self.pre
            y = torch.einsum('...i,ai,ab,bo->...o', x, self.A, self.M, self.B)
            if postmult:
                y = y * self.post
            return self.linear(x) + y
    return functools.partial(LinearWithTiedLoraExtra, A=A, B=B)

def get_tied_lora_extra_model(model,a=32,b=32,premult=False,postmult=True):
    return wrap_like_linear(model, functools.partial(create_tied_lora_extra_wrapper, a=b, b=b, 
                                                     premult=premult, postmult=postmult, dtype=model.dtype))

def get_lora_model(model,rank=8):
    class LinearWithLoRA(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            self.linear = linear
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = linear.weight.dtype))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = linear.weight.dtype ))
        def forward(self, x):
            return self.linear(x) + x @ self.A @ self.B
    return wrap_linear(model,functools.partial(LinearWithLoRA, rank=rank))

def get_dora_model(model,rank=8):
    class LinearWithDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = linear.weight.dtype))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = linear.weight.dtype))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + self.A @ self.B, dim=0, keepdim=True)
            return (x @ self.W + x @ self.A @ self.B) * self.mag / norm
            # W = self.W + self.A @ self.B
            # W *= self.mag / torch.linalg.norm(W, dim=0, keepdim=True)
            # return x @ W
    return wrap_linear(model,functools.partial(LinearWithDora, rank=rank))

def get_simple_dora_model(model,rank=8):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = linear.weight.dtype))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = linear.weight.dtype))
        def forward(self, x):
            return (x @ self.W + x @ self.A @ self.B) * self.mag
    return wrap_linear(model,functools.partial(LinearWithSimpleDoraTranspose, rank=rank))

def get_dora_transpose_model(model,rank=8):
    class LinearWithTransposeDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = linear.weight.dtype))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = linear.weight.dtype))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + self.A @ self.B, dim=1, keepdim=True)
            x *= (self.mag / norm).view(-1)
            return x @ self.W + x @ self.A @ self.B
    return wrap_linear(model,functools.partial(LinearWithTransposeDora, rank=rank))

def get_simple_dora_transpose_model(model,rank=8):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = linear.weight.dtype))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = linear.weight.dtype))
        def forward(self, x):
            x *= self.mag.view(-1)
            return x @ self.W + x @ self.A @ self.B
    return wrap_linear(model,functools.partial(LinearWithSimpleDoraTranspose, rank=rank))

def get_simple_svdora_model(model,rankU=8, rankV=8):
    class LinearWithSimpleSvdora(torch.nn.Module):
        def __init__(self, linear, rankU, rankV):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            print('here', W.shape)
            U, sigma, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
            print(sigma)
            self.U = torch.nn.Parameter(U.to(W.dtype), requires_grad=False)
            self.sigma = torch.nn.Parameter(sigma.to(W.dtype))
            self.Vh = torch.nn.Parameter(Vh.to(W.dtype), requires_grad=False)
            
            self.A1 = torch.nn.Parameter(torch.randn(U.shape[0], rankU, dtype = linear.weight.dtype))
            self.B1 = torch.nn.Parameter(torch.zeros(rankU, U.shape[1], dtype = linear.weight.dtype))
            self.A2 = torch.nn.Parameter(torch.randn(Vh.shape[0], rankV, dtype = linear.weight.dtype))
            self.B2 = torch.nn.Parameter(torch.zeros(rankV, Vh.shape[1], dtype = linear.weight.dtype))
        def forward(self, x):
            x = x @ self.U + x @ self.A1 @ self.B1
            x = x * self.sigma
            return x @ self.Vh + x @ self.A2 @ self.B2
    return wrap_linear(model,functools.partial(LinearWithSimpleSvdora, rankU=rankU, rankV=rankV))
