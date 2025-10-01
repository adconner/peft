import torch
import numpy as np
import functools

from dataclasses import dataclass
import draccus

@dataclass
class PeftStrategyConfig(draccus.ChoiceRegistry):
    pass

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
def create_tensor_embedding_wrapper(in_features, out_features, a, b, l, premult, postmult, dtype, alpha, gamma):
    T = torch.nn.Parameter(torch.randn(a, b, l, dtype=dtype) / gamma)
    A = torch.nn.Parameter(torch.randn(a, in_features, dtype=dtype) / gamma)
    B = torch.nn.Parameter(torch.zeros(b, out_features, dtype=dtype))
    class LinearWithTensorEmbedding(torch.nn.Module):
        def __init__(self, linear, T, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.T = T
            self.A = A
            self.B = B
            self.M = torch.nn.Parameter(torch.randn(l, dtype=dtype) / gamma)
            self.norm = float(np.sqrt(self.linear.weight.float().pow(2).mean()))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            return self.linear(x) + gamma**3 * alpha * self.norm / np.sqrt(a*b*l) *\
                    torch.einsum('...i,ai,bo,l,abl,i,o->...o', x, self.A, self.B, self.M, self.T,
                                                 self.pre if premult else torch.ones((1,),dtype=dtype),
                                                 self.post if postmult else torch.ones((1,),dtype=dtype))
    return functools.partial(LinearWithTensorEmbedding, T=T, A=A, B=B)
@PeftStrategyConfig.register_subclass('tensor_embedding')
@dataclass
class TensorEmbeddingConfig(PeftStrategyConfig):
    a: int = 16
    b: int = 16
    l: int = 8
    premult: bool = False
    postmult: bool = False
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        return wrap_like_linear(model, functools.partial(create_tensor_embedding_wrapper, a=self.a, b=self.b, l=self.l, 
                                                     premult=self.premult, postmult=self.postmult, dtype=model.dtype,
                                                         alpha=self.alpha, gamma=self.gamma))
    
def create_tied_lora_extra_wrapper(in_features, out_features, a, b, premult, postmult, dtype, alpha, gamma):
    A = torch.nn.Parameter(torch.randn(a, in_features, dtype=dtype) / gamma)
    B = torch.nn.Parameter(torch.zeros(b, out_features, dtype=dtype))
    class LinearWithTiedLoraExtra(torch.nn.Module):
        def __init__(self, linear, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.A = A
            self.B = B
            self.M = torch.nn.Parameter(torch.randn(a, b, dtype=dtype) / gamma)
            self.norm = float(np.sqrt(self.linear.weight.float().pow(2).mean()))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            return self.linear(x) + gamma**2 * alpha * self.norm / np.sqrt(a*b) *\
                    torch.einsum('...i,ai,ab,bo,i,o->...o', x, self.A, self.M, self.B,
                                                 self.pre if premult else torch.ones((1,),dtype=dtype),
                                                 self.post if postmult else torch.ones((1,),dtype=dtype))
    return functools.partial(LinearWithTiedLoraExtra, A=A, B=B)
@PeftStrategyConfig.register_subclass('tied_lora_extra')
@dataclass
class TiedLoraExtraConfig(PeftStrategyConfig):
    a: int = 16
    b: int = 16
    premult: bool = False
    postmult: bool = False
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        return wrap_like_linear(model, functools.partial(create_tied_lora_extra_wrapper, a=self.a, b=self.b, 
                                                     premult=self.premult, postmult=self.postmult, 
                                                     dtype=model.dtype, alpha=self.alpha, gamma=self.gamma))
    
def create_tied_lora_wrapper(in_features, out_features, r, premult, postmult, dtype, alpha, gamma):
    A = torch.nn.Parameter(torch.randn(in_features, r, dtype=dtype) / gamma)
    B = torch.nn.Parameter(torch.zeros(r, out_features, dtype=dtype))
    class LinearWithTiedLoraExtra(torch.nn.Module):
        def __init__(self, linear, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.A = A
            self.B = B
            self.M = torch.nn.Parameter(torch.randn(r, dtype=dtype) / gamma)
            self.norm = float(np.sqrt(self.linear.weight.float().pow(2).mean()))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            return self.linear(x) + gamma**2 * alpha * self.norm / np.sqrt(r) *\
                    torch.einsum('...i,ir,r,ro,i,o->...o', x, self.A, self.M, self.B,
                                                 self.pre if premult else torch.ones((1,),dtype=dtype),
                                                 self.post if postmult else torch.ones((1,),dtype=dtype))
    return functools.partial(LinearWithTiedLoraExtra, A=A, B=B)
@PeftStrategyConfig.register_subclass('tied_lora')
@dataclass
class TiedLoraConfig(PeftStrategyConfig):
    r: int = 16
    premult: bool = False
    postmult: bool = True
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        return wrap_like_linear(model, functools.partial(create_tied_lora_wrapper, r=self.r,
                                                 premult=self.premult, postmult=self.postmult, 
                                                 dtype=model.dtype, alpha=self.alpha, gamma=self.gamma))

def create_partially_tied_lora_wrapper(in_features, out_features, r, la, lb, premult, midmult, postmult, dtype, alpha, gamma):
    A = torch.nn.Parameter(torch.randn(in_features, r, la, dtype=dtype) / gamma)
    B = torch.nn.Parameter(torch.zeros(r, out_features, lb, dtype=dtype))
    class LinearWithTiedLoraExtra(torch.nn.Module):
        def __init__(self, linear, A, B):
            super().__init__()
            assert linear.bias is None
            assert linear.weight.dtype is dtype
            self.linear = linear
            self.A = A
            self.B = B
            self.MA = torch.nn.Parameter(torch.randn(la, dtype=dtype) / gamma)
            self.MB = torch.nn.Parameter(torch.randn(lb, dtype=dtype) / gamma)
            self.norm = float(np.sqrt(self.linear.weight.float().pow(2).mean()))
            if premult:
                self.pre = torch.nn.Parameter(torch.ones(in_features,dtype=dtype))
            if midmult:
                self.mid = torch.nn.Parameter(torch.ones(r,dtype=dtype))
            if postmult:
                self.post = torch.nn.Parameter(torch.ones(out_features,dtype=dtype))
        def forward(self, x):
            return self.linear(x) + gamma ** 3 * alpha * self.norm / np.sqrt(r*la*lb) *\
                    torch.einsum('...i,irA,roB,A,B,i,r,o->...o', x, self.A, self.B, self.MA, self.MB, 
                             self.pre if premult else torch.ones((1,),dtype=dtype),
                             self.mid if midmult else torch.ones((1,),dtype=dtype),
                             self.post if postmult else torch.ones((1,),dtype=dtype))
    return functools.partial(LinearWithTiedLoraExtra, A=A, B=B)
@PeftStrategyConfig.register_subclass('partially_tied_lora')
@dataclass
class PartiallyTiedLoraConfig(PeftStrategyConfig):
    r: int = 8
    la: int = 4
    lb: int = 4
    premult: bool = False
    midmult: bool = False
    postmult: bool = False
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        return wrap_like_linear(model, functools.partial(create_partially_tied_lora_wrapper, r=self.r, la=self.la, lb=self.lb,
                                                 premult=self.premult, midmult=self.midmult, postmult=self.postmult, 
                                                 dtype=model.dtype, alpha=self.alpha, gamma=self.gamma))

@PeftStrategyConfig.register_subclass('lora')
@dataclass
class LoraConfig(PeftStrategyConfig):
    r: int = 8
    alpha: float = 3.
    gamma: float = 1.
    def wrap(self,model):
        r = self.r
        alpha = self.alpha
        gamma = self.gamma
        class LinearWithLoRA(torch.nn.Module):
            def __init__(self, linear, r, alpha):
                super().__init__()
                assert linear.bias is None
                self.linear = linear
                self.alpha = alpha
                self.A = torch.nn.Parameter(torch.randn(linear.in_features, r, dtype=linear.weight.dtype) / gamma)
                self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features, dtype=linear.weight.dtype))
            def forward(self, x):
                return self.linear(x) + gamma * alpha / np.sqrt(r) * (x @ self.A @ self.B)
        return wrap_linear(model,functools.partial(LinearWithLoRA, r=self.r, alpha=self.alpha))
    
@PeftStrategyConfig.register_subclass('strong_gamma_lora')
@dataclass
class StrongGammaLoraConfig(PeftStrategyConfig):
    r: int = 8
    alpha: float = 3.
    gamma: float = 750.
    def wrap(self,model):
        r = self.r
        alpha = self.alpha
        gamma = self.gamma
        class LinearWithLoRA(torch.nn.Module):
            def __init__(self, linear, r, alpha):
                super().__init__()
                assert linear.bias is None
                self.linear = linear
                self.alpha = alpha
                self.A = torch.nn.Parameter(torch.randn(linear.in_features, r, dtype=linear.weight.dtype) / gamma)
                self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features, dtype=linear.weight.dtype))
            def forward(self, x):
                return self.linear(x) + torch.minimum(torch.tensor(gamma).type_as(self.B), self.B.pow(2).mean().rsqrt()).detach() * \
                        alpha / np.sqrt(r) * (x @ self.A @ self.B)
        return wrap_linear(model,functools.partial(LinearWithLoRA, r=self.r, alpha=self.alpha))
    
@PeftStrategyConfig.register_subclass('normed_lora')
@dataclass
class NormedLoraConfig(PeftStrategyConfig):
    r: int = 8
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        r = self.r
        alpha = self.alpha
        gamma = self.gamma
        class LinearWithLoRA(torch.nn.Module):
            def __init__(self, linear, r, alpha):
                super().__init__()
                assert linear.bias is None
                self.linear = linear
                self.alpha = alpha
                self.norm = float(np.sqrt(self.linear.weight.float().pow(2).mean()))
                self.A = torch.nn.Parameter(torch.randn(linear.in_features, r, dtype=linear.weight.dtype) / gamma)
                self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features, dtype=linear.weight.dtype))
            def forward(self, x):
                return self.linear(x) + gamma * alpha * self.norm / np.sqrt(r) * (x @ self.A @ self.B)
        return wrap_linear(model,functools.partial(LinearWithLoRA, r=self.r, alpha=self.alpha))

@PeftStrategyConfig.register_subclass('dora')
@dataclass
class DoraConfig(PeftStrategyConfig):
    r: int = 8
    transpose: bool = False
    eps: float = 1e-6
    alpha: float = 100.
    gamma: float = 1.
    def wrap(self,model):
        # ordinary dora scales the input (so reduces in the output dimension)
        # we fix input dimension as 0 and output dimension as 1 to agree with matrices acting on the right
        transpose = self.transpose
        reducedim = 1 if self.transpose else 0
        scaledim_name = 'i' if transpose else 'o'
        eps = self.eps
        r = self.r
        alpha = self.alpha
        gamma = self.gamma
        class LinearWithDora(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                assert linear.bias is None
                W = linear.weight.T
                mag = torch.sqrt(W.float().pow(2).mean(dim=reducedim) + eps).type_as(W)
                self.W = torch.nn.Parameter(W.contiguous() / float(np.sqrt(W.float().pow(2).mean())),
                                            requires_grad = False)
                self.mag = torch.nn.Parameter(mag)
                self.A = torch.nn.Parameter(torch.randn(linear.in_features, r, dtype = W.dtype) / gamma)
                self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features, dtype = W.dtype))
            def forward(self, x):
                Wtune = self.W + gamma * alpha / np.sqrt(r) * self.A @ self.B
                imag = torch.rsqrt(Wtune.pow(2).mean(dim=reducedim) + eps)
                return torch.einsum(f'...i,io,{scaledim_name}->...o', x, Wtune, imag * self.mag)
        return wrap_linear(model,functools.partial(LinearWithDora))

@PeftStrategyConfig.register_subclass('simple_dora')
@dataclass
class SimpleDoraConfig(PeftStrategyConfig):
    r: int = 8
    transpose: bool = False
    eps: float = 1e-6
    alpha: float = 100.
    beta: float = 1. # mag learning rate boost
    gamma: float = 750.
    def wrap(self,model):
        transpose = self.transpose
        eps = self.eps
        reducedim = 1 if self.transpose else 0
        r = self.r
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        class LinearWithSimpleDoraTranspose(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                assert linear.bias is None
                W = linear.weight.T
                mag = torch.sqrt(W.float().pow(2).mean(dim=reducedim,keepdim=True) + eps)
                self.W = torch.nn.Parameter((W.float() / mag).type_as(W).contiguous(), requires_grad = False)
                self.imag = torch.nn.Parameter(mag.type_as(W).squeeze() * beta,requires_grad=False)
                self.mag = torch.nn.Parameter(torch.ones(self.imag.shape,dtype=W.dtype) / beta)
                self.A = torch.nn.Parameter(torch.randn(linear.in_features, r, dtype = W.dtype) / gamma)
                self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features, dtype = W.dtype))
            def forward(self, x):
                if transpose:
                    x = x * self.imag * self.mag
                y = x @ self.W + gamma * alpha / np.sqrt(r) * x @ self.A @ self.B
                if not transpose:
                    y = y * self.imag * self.mag
                return y
        return wrap_linear(model,functools.partial(LinearWithSimpleDoraTranspose))

@PeftStrategyConfig.register_subclass('svdora')
@dataclass
class SvdoraConfig(PeftStrategyConfig):
    rU: int = 8
    rV: int = 8
    alpha: float = 100.
    gamma: float = 750.
    def wrap(self,model):
        alpha = self.alpha
        gamma = self.gamma
        rU = self.rU
        rV = self.rV
        class LinearWithSimpleSvdora(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                assert linear.bias is None
                W = linear.weight.T
                U, sigma, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
                print(sigma)
                self.U = torch.nn.Parameter(U.to(W.dtype).contiguous(), requires_grad=False)
                self.sigma = torch.nn.Parameter(sigma.to(W.dtype))
                self.Vh = torch.nn.Parameter(Vh.to(W.dtype).contiguous(), requires_grad=False)
                
                self.A1 = torch.nn.Parameter(torch.randn(U.shape[0], rU, dtype = linear.weight.dtype)/gamma)
                self.B1 = torch.nn.Parameter(torch.zeros(rU, U.shape[1], dtype = linear.weight.dtype))
                self.A2 = torch.nn.Parameter(torch.randn(Vh.shape[0], rV, dtype = linear.weight.dtype)/gamma)
                self.B2 = torch.nn.Parameter(torch.zeros(rV, Vh.shape[1], dtype = linear.weight.dtype))
            def forward(self, x):
                x = x @ self.U + gamma * alpha / np.sqrt(rU * self.U.shape[1]) * (x @ self.A1 @ self.B1)
                x = x * self.sigma
                return x @ self.Vh + gamma * alpha / np.sqrt(rV * self.Vh.shape[0]) * x @ self.A2 @ self.B2
        return wrap_linear(model,functools.partial(LinearWithSimpleSvdora))
