import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from nn import Linear, Tanh, BatchNorm1d
# set up the data
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

itos = {i+1: s for i, s in enumerate(chars)}
itos[0] = '.'
stoi = {s: i for i,s in itos.items()}


context_size = 4

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * context_size
        for ch in w + '.':
            
            # decode into a number 
            ix = stoi[ch]
            # this is the input
            X.append(context)
            # this is the target
            Y.append(ix)

            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# for splitting the dataset into train, val, test
random.shuffle(words)
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

xtr, ytr = build_dataset(words[:n1])
xval, yval = build_dataset(words[n1:n2])
xtest, ytest = build_dataset(words[n2:])


# initialize the weights 
n_hidden = 300
embedding_length = 10
vocab_size = 27
batch_size = 64

C = torch.randn((vocab_size, embedding_length)) # 27 chars, vectors of length 10 
#W1 = torch.randn((embedding_length*context_size, n_hidden)) * 0.2
#b1 = torch.randn((n_hidden)) * 0.01
#W2 = torch.randn((n_hidden, vocab_size)) * 0.01
#b2 = torch.randn((vocab_size)) * 0

C = torch.randn((27, embedding_length))
layers = [
    Linear(embedding_length*4, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, 27),
    BatchNorm1d(27)
]

with torch.no_grad():
    # reduce confidence to lower initial loss
    layers[-1].gamma *= 0.1
    # apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad_()


# track stats
lossi = []
stepi = []
max_steps = 250000
for i in range(max_steps):
    ix = torch.randint(0, xtr.shape[0], (batch_size, ))
    emb = C[xtr[ix]]
    x = emb.view(64, -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, ytr[ix])

    # reset gradient
    for p in parameters:
        p.grad = None

    loss.backward()
    lr = 0.1 if i < 125000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad
    
    
    
# for testing loss 
@torch.no_grad()
def test_splits(split):
    x, y = {
        "train": (xtr, ytr),
        "val": (xval, yval),
        "test": (xtest, ytest)
    }[split]
    emb = C[x]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(f"{split} loss: {loss.item()}")


test_splits("train")   
test_splits("val")
test_splits("test") 

# sample from the model

for _ in range(20):
    out = []
    context = [0] * context_size
    while True:
        
        # one example at a time
        emb = C[context]
        x = emb.view(1, -1)
        for layer in layers:
            layer.training = False
            x = layer(x)
        
        
        probs = F.softmax(x)
        ix = torch.multinomial(probs, num_samples=1).item()

        char = itos[ix]
        out.append(char)

        if char == '.':
            break
        context = context[1:] + [ix]

    print(''.join(out))