from data_import import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from memory_exp import *

device = 'cuda:0'

def gram_schmidt(vectors):
    num_vectors, dim = vectors.shape
    ortho_vectors = np.zeros((num_vectors, dim))

    for i in range(num_vectors):
        ortho_vectors[i] = vectors[i]

        for j in range(i):
            ortho_vectors[i] -= np.dot(vectors[i], ortho_vectors[j]) / np.dot(ortho_vectors[j], ortho_vectors[j]) * ortho_vectors[j]

    # Normalize the vectors
    ortho_vectors = np.array([v / np.linalg.norm(v) for v in ortho_vectors])

    return ortho_vectors

def generate_orthogonal_vectors(n):
    random_vectors = np.random.rand(n, n)  # Generate random vectors
    orthogonal_vectors = gram_schmidt(random_vectors)

    return orthogonal_vectors

def relu(x):
    return np.where(x<0,0,x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def implement_W2(interval):
    L = len(interval)
    # if the interval is descending, we need to reverse it
    reverted = False
    if interval[0] > interval[-1]:
        interval = interval[::-1]
        reverted = True
    ws = np.zeros(L+1)
    bs = np.zeros(L+1)
    ws[0] = -3
    bs[0] = 30
    for i, v in enumerate(interval):
        i = i+1
        ws[i] = ws[i-1] + 0.5
        bs[i] = (v * ws[i-1] + bs[i-1]) - (ws[i] * v)
        
    if reverted:
        ws = ws[::-1]
        bs = bs[::-1]
    return ws, bs

def test(model, T , L, has_BOS = True, verbose=False):
    works = True
    for k in range(1,L+1):
        if has_BOS:
            x = [T] + [np.random.randint(1,T)] * k + (L-k) * [0]
        else:
            x = [np.random.randint(1,T)] * k + (L-k) * [0]
        x = torch.tensor(x ).to(device).reshape(1,-1)
        r = model(x)
        if has_BOS:
            x_hat = r[0,1].argmax().item()
        else:
            x_hat = r[0,0].argmax().item()
        if x_hat != k:
            if verbose:
                print(f"I got {x_hat}, but should have been {k}.")
            works = False
    if works:
        print("Test passed, model works!")
    else:
        print(":((( Test failed, model does not work!")

def generate_perfect_weights(config):
    if config['attention_input'] == 'only_sem' and config['no_softmax'] == False and config['dataset_type'] == 'backward':
        return generate_perfect_weights_dot_sftm(config)
    elif config['attention_input'] == 'linear' and config['no_softmax'] == False and config['dataset_type'] == 'backward':
        return generate_perfect_weights_linear_sftm(config)
    elif config['attention_input'] == 'linear' and config['no_softmax'] == True and config['dataset_type'] == 'backward':
        return generate_perfect_weights_linear(config)
    elif config['attention_input'] == 'only_sem' and config['no_softmax'] == False and config['dataset_type'] == 'backward_BOS':
        return generate_perfect_weights_bos(config)
    elif config['attention_input'] == 'only_sem' and config['no_softmax'] == True and config['dataset_type'] == 'backward_BOS':
        return generate_perfect_weights_bos_nsftm(config)
    elif config['attention_input'] == 'only_sem' and config['no_softmax'] == True and config['dataset_type'] == 'backward':
        """Backward counter"""
        return generate_perfect_weights_dot_nosftm(config)
    else:
        raise ValueError(f"Cannot generate perfect weights for {config['attention_input']}")
    
def generate_perfect_weights_dot_nosftm(config):
    assert config['T'] == config['model_dim']
    assert config['p'] == 1
    
    L = config['seq_len']
    T = config['T']
    has_BOS =  'BOS' in config['dataset_type']
    
    

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                            model_dim=config['model_dim'],
                                            p=config['p'],n_classes=config['seq_len']+1,
                                            L=config['seq_len'],
                                            attention_input=config['attention_input'],use_softmax=not config['no_softmax']).to(device)

    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])
    counter = word_embeddings_basis.sum(axis=0)
    word_embeddings = word_embeddings_basis + counter[None,:]
    
    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)

    model.token_mixer.Q.data = torch.eye(config['model_dim']).to(device) * np.sqrt(config['model_dim'])
    model.token_mixer.K.data = torch.eye(config['model_dim']).to(device)
    
    W_1 = counter.reshape(-1,1).repeat(1,1).T  / (T+1)
    b_1 = 0
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)
    
    softmax_table = np.arange(1,L+1) + L*(T+2)+1
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2 #* 30

    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(1,1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries
    
    
def generate_perfect_weights_bos(config):
    assert config['T'] == config['model_dim']
    assert config['p'] == 1
    T = config['T']
    
    L = config['seq_len']
    has_BOS =  'BOS' in config['dataset_type']
    

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                            model_dim=config['model_dim'],
                                            p=config['p'],n_classes=config['seq_len']+1,
                                            L=config['seq_len'],
                                            attention_input=config['attention_input'],use_softmax=not config['no_softmax']).to(device)

    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])
    BOS = word_embeddings_basis.sum(axis=0)
    word_embeddings = np.concatenate([word_embeddings_basis, BOS[None,:]])

    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)

    model.token_mixer.Q.data = torch.eye(config['model_dim']).to(device) * np.sqrt(config['model_dim'])
    model.token_mixer.K.data = torch.eye(config['model_dim']).to(device)

    softmax_table = []
    for i in range(1,config['seq_len']+1):
        x = [T] + [0] * i + (L-i) * [1]
        x = torch.tensor(x ).to(device).reshape(1,-1)
        # load attn matrix
        r = model(x)
        attn_probs = model.attn_probs.cpu().detach().numpy()
        print(i, attn_probs[0,1,0], attn_probs[0,1,i])
        softmax_table.append(attn_probs[0,1,0] * (T - 1) + 2)

    W_1 = BOS.reshape(-1,1).repeat(1,1).T 
    b_1 = 0
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)
    
    softmax_table = np.array(softmax_table)
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2 #* 30

    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(1,1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries

def generate_perfect_weights_bos_nsftm(config):
    assert config['T'] == config['model_dim']
    assert config['p'] == 1
    T = config['T']
    
    L = config['seq_len']
    has_BOS =  'BOS' in config['dataset_type']
    

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                            model_dim=config['model_dim'],
                                            p=config['p'],n_classes=config['seq_len']+1,
                                            L=config['seq_len'],
                                            attention_input=config['attention_input'],use_softmax=not config['no_softmax']).to(device)

    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])
    BOS = word_embeddings_basis.sum(axis=0)
    word_embeddings = np.concatenate([word_embeddings_basis, BOS[None,:]])

    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)

    model.token_mixer.Q.data = torch.eye(config['model_dim']).to(device) * np.sqrt(config['model_dim'])
    model.token_mixer.K.data = torch.eye(config['model_dim']).to(device)

    softmax_table = []
    for i in range(1,config['seq_len']+1):
        x = [T] + [0] * i + (L-i) * [1]
        x = torch.tensor(x ).to(device).reshape(1,-1)
        # load attn matrix
        r = model(x)
        attn_probs = model.attn_probs.cpu().detach().numpy()
        softmax_table.append(attn_probs[0,1,0] * T + attn_probs[0,1,1:i+1].sum())

    W_1 = BOS.reshape(-1,1).repeat(1,1).T 
    b_1 = -1.0
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)
    
    softmax_table = np.array(softmax_table)
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2 #* 30

    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(1,1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries

def generate_perfect_weights_linear_sftm(config):
    assert config['T'] == config['model_dim'] == config['p']
    has_BOS =  'BOS' in config['dataset_type']
    L = config['seq_len'] 

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                model_dim=config['model_dim'],
                                p=config['p'],n_classes=config['seq_len']+1,
                                L=config['seq_len'],
                                attention_input=config['attention_input'],
                                use_softmax=not config['no_softmax']).to(device)
    
    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])

    word_embeddings = word_embeddings_basis 
    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)
    
    a = 1/(config['seq_len']+1)
    A = a * np.ones((config['seq_len'],config['seq_len'])) + np.eye(config['seq_len']) * a
    model.token_mixer.A.data = torch.tensor(A).float().to(device)
    
    softmax_table = []
    for i in range(1,config['seq_len']+1):
        x = [0] * i + (L-i) * [1]
        x = torch.tensor(x ).to(device).reshape(1,-1)
        # load attn matrix
        r = model(x)
        attn_probs = model.attn_probs.cpu().detach().numpy()
        softmax_table.append(attn_probs[0,0,0:i].sum())
 
    W_1 = torch.tensor(word_embeddings).float().to(device)
    b_1 = - 1.0
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)
    softmax_table = np.array(softmax_table)
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2
    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(config['T'],1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries

   
def generate_perfect_weights_linear(config):
    assert config['T'] == config['model_dim'] == config['p']
    has_BOS =  'BOS' in config['dataset_type']
    L = config['seq_len'] 

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                model_dim=config['model_dim'],
                                p=config['p'],n_classes=config['seq_len']+1,
                                L=config['seq_len'],
                                attention_input=config['attention_input'],
                                use_softmax=not config['no_softmax']).to(device)
    
    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])

    word_embeddings = word_embeddings_basis 
    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)
    
    a = 1/(config['seq_len']+1)
    
    A = a * np.ones((config['seq_len'],config['seq_len'])) + np.eye(config['seq_len']) * a
    
    model.token_mixer.A.data = torch.tensor(A).float().to(device)
    W_1 = torch.tensor(word_embeddings).float().to(device)
    b_1 = - (1.0+a)
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)

    
    softmax_table = np.cumsum(np.ones(config['seq_len']+1) * a)
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2
    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(config['T'],1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries
        
def generate_perfect_weights_dot_sftm(config):
    assert config['T'] == config['model_dim'] == config['p']
    
    L = config['seq_len']
    has_BOS =  'BOS' in config['dataset_type']

    model = TransformerSeq2Seq(T=config['T']+int(has_BOS),
                                model_dim=config['model_dim'],
                                p=config['p'],n_classes=config['seq_len']+1,
                                L=config['seq_len'],
                                attention_input=config['attention_input'],
                                use_softmax=not config['no_softmax']).to(device)

    word_embeddings_basis = generate_orthogonal_vectors(config['model_dim'])

    word_embeddings = word_embeddings_basis 
    model.semantic_emb.weight.data = torch.tensor(word_embeddings).float().to(device)

    model.token_mixer.Q.data = torch.eye(config['model_dim']).to(device) * np.sqrt(config['model_dim'])
    model.token_mixer.K.data = torch.eye(config['model_dim']).to(device)

    softmax_table = []
    for i in range(1,config['seq_len']+1):
        x = [0] * i + (L-i) * [1]
        x = torch.tensor(x ).to(device).reshape(1,-1)
        # load attn matrix
        r = model(x)
        attn_probs = model.attn_probs.cpu().detach().numpy()
        softmax_table.append(attn_probs[0,0,0]*i)
    
    W_1 = torch.tensor(word_embeddings).float().to(device)
    b_1 = -1.0
    model.fc1.weight.data = torch.tensor(W_1).float().to(device)
    model.fc1.bias.data = torch.tensor(b_1).float().to(device)
    
    softmax_table = np.array(softmax_table)
    descision_boundaries = softmax_table[:-1] + (softmax_table[1:] - softmax_table[:-1])/2
    ws, bs = implement_W2(descision_boundaries)
    # weight 0 is always smallest
    ws = np.array([-100] + list(ws))
    bs = np.array([-100] + list(bs))
    W_2, b_2 = ws, bs
    W_2 = W_2[:,None].repeat(config['T'],1)
    model.fc2.weight.data = torch.tensor(W_2).float().to(device)
    model.fc2.bias.data = torch.tensor(b_2).float().to(device)
    return model, ws, bs, softmax_table, descision_boundaries