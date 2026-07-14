import torch 


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) * 1/(fan_in ** 2)
        self.bias = torch.zeros(fan_out) if True else None
    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    def parameters(self):
        return [self.weight] + ([] if self.bias is  None else [self.bias])       


class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    def parameters(self):
        return []


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        # for the running std and mean averages
        self.momentum=momentum
        self.training = True
        # parameters use backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers, trained with running mean
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/(xvar + self.eps)**0.5 
        self.out = self.gamma * xhat + self.beta
        # if training update the buffers 
        if self.training:
            with torch.no_grad():
                self.running_mean = ((1-self.momentum) 
                                    * self.running_mean + self.momentum * xmean)
                self.running_var = ((1-self.momentum) * 
                                    self.running_var + self.momentum*xvar) 
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]
    