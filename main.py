import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
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
W1 = torch.randn((embedding_length*context_size, n_hidden)) * 0.2
b1 = torch.randn((n_hidden)) * 0.01
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
b2 = torch.randn((vocab_size)) * 0

parameters = [C, W1, b1, W2, b2]
# track grad for backwards pass 
for p in parameters:
    p.requires_grad_()

# track stats
lossi = []
stepi = []
max_steps = 250000
for i in range(max_steps):
    ix = torch.randint(0, xtr.shape[0], (batch_size, ))
    emb = C[xtr[ix]]
    emb = emb.view(64, -1)
    preact = emb @ W1 + b1 
    act = torch.tanh(preact)
    logits = act @ W2 + b2
    loss = F.cross_entropy(logits, ytr[ix])
    if i == 0:
        print(loss.item())
    for p in parameters:
        p.grad = None
    loss.backward()
    # use a decaying learning rate
    lr = 0.1 if i < 120000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.item())
    stepi.append(i)
    
# for testing end loss 
def test_splits(split):
    x, y = {
        "train": (xtr, ytr),
        "val": (xval, yval),
        "test": (xtest, ytest)
    }[split]
    emb = C[x]
    emb = emb.view(emb.shape[0], -1)
    act = torch.tanh(emb @ W1 + b1)
    logits = act @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f"{split} loss: {loss:.4f}")

test_splits("train")   
test_splits("val")
test_splits("test") 