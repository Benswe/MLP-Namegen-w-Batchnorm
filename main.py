words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

itos = {i+1: s for i, s in enumerate(chars)}
itos[0] = '.'
stoi = {s: i for i,s in itos.items()}
