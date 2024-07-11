
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt
import wandb
import argparse


device = 'cuda:0'

def hist(s):
  c = Counter(s)
  c = {w: c[w] for w in c}
  return [c[w] for w in s]

class HistogramDatasetSampleForward(Dataset):
    has_BOS = False
    def __init__(self, seq_len, T, n_samples,seed=None):
        self.seed = seed if seed is not None else np.random.randint(0,100000)
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        rs = np.random.RandomState(self.seed)
        self.X = rs.randint(0, T, (n_samples, seq_len))
        self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = hist(self.X[i])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)

class ExclusionDataset:
  def __init__(self,validation):
    data = np.unique(validation, axis=0)
    # Get the indices that would sort the rows lexicographically
    sorted_indices = np.lexsort(data.T)
    # Use the sorted indices to rearrange the original array
    self.data = np.take(data, sorted_indices, axis=0)
    
  def check_sample(self,sample):
    data = self.data
    idx_left = 0
    idx_right = data.shape[0]
    for i in reversed(range(data.shape[1])):
        idx_left_ = np.searchsorted(data[idx_left:idx_right,i],sample[i],side='left')
        idx_right_ = np.searchsorted(data[idx_left:idx_right,i],sample[i],side='right')
        idx_left_= idx_left + idx_left_
        idx_right_ = idx_left + idx_right_ 
        idx_left = idx_left_
        idx_right = idx_right_
    return idx_left < data.shape[0] and np.array_equal(sample,data[idx_left]), idx_left
    
  def __contains__(self, x):
    return self.check_sample(x)[0]
    
class HistogramDatasetSampleBackwardReverse(Dataset):
  
    has_BOS = False
    def __init__(self, seq_len, T, n_samples,seed=None,validation=None):
        self.seed = seed if seed is not None else np.random.randint(0,100000)
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        self.rs = np.random.RandomState(self.seed)
        
        if validation is None:
          data = [self.random_sequence(seq_len,T) for _ in range(n_samples)]
        else:
          data = []
          for i in range(n_samples):
            x = self.random_sequence(seq_len,T)
            if x not in validation:
              data.append(x)
          #print(len(data),n_samples)
              
        self.X = np.vstack(data)
        #self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = np.array(hist(self.X[i])[::-1])

    def random_sequence(self,n,T):
      partition = self.random_partition(n)
      partition = np.cumsum(partition)
      if len(partition) > T:
        partition = partition[:T-1] + sum(partition[T-1:])
      tokens = self.rs.choice(range(T),size=len(partition),replace=False)
      sequence = np.zeros(n)
      sequence[:partition[0]] = tokens[0]
      for i,j,t in zip(partition[:-1],partition[1:],tokens[1:]):
        sequence[i:j] = t
      self.rs.shuffle(sequence)
      return np.array(sequence)

    def random_partition(self,n):
      if n == 0:
        return []
      if n == 1:
        return [1]
      k = self.rs.randint(1,n)
      return [k] + self.random_partition(n-k)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)
      

class HistogramDatasetSampleBackwardEmbd(Dataset):
    has_BOS = False
    def __init__(self, seq_len, T, n_samples,seed=None,validation=None,embedding_dim=100,eps=0.001, mus=None):
        self.seed = seed if seed is not None else np.random.randint(0,100000)
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        self.rs = np.random.RandomState(self.seed)
        self.embedding_dim = embedding_dim
        
        self.eps = eps
        self.factor = 1/np.sqrt(1+eps**2)
        if mus is None:
          self.mus = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(T, embedding_dim))
        else: 
          self.mus = mus
        
        if validation is None:
          data = [self.random_sequence(seq_len,T) for _ in range(n_samples)]
        else:
          data = []
          for i in range(n_samples):
            x = self.random_sequence(seq_len,T)
            if x not in validation:
              data.append(x)
          #print(len(data),n_samples)
              
        self.X = np.vstack(data)
        #self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = np.array(hist(self.X[i]))
          
        # change dtype of X to int
        self.X = self.X.astype(np.int32)

    def random_sequence(self,n,T):
      partition = self.random_partition(n)
      partition = np.cumsum(partition)
      if len(partition) > T:
        partition = partition[:T-1] + sum(partition[T-1:])
      tokens = self.rs.choice(range(T),size=len(partition),replace=False)
      sequence = np.zeros(n)
      sequence[:partition[0]] = tokens[0]
      for i,j,t in zip(partition[:-1],partition[1:],tokens[1:]):
        sequence[i:j] = t
      self.rs.shuffle(sequence)
      return np.array(sequence)

    def random_partition(self,n):
      if n == 0:
        return []
      if n == 1:
        return [1]
      k = self.rs.randint(1,n)
      return [k] + self.random_partition(n-k)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # add randomness
        nu = np.random.normal(loc=0, scale=np.sqrt(1/self.embedding_dim), size=(self.seq_len+int(self.has_BOS),self.embedding_dim))
        data = (self.mus[self.X[idx]] + self.eps * nu) * self.factor
        return torch.tensor(data,dtype=torch.float32), torch.tensor(self.y[idx],dtype=torch.long)
  

class HistogramDatasetSampleBackward(Dataset):
  
    has_BOS = False
    def __init__(self, seq_len, T, n_samples,seed=None,validation=None):
        self.seed = seed if seed is not None else np.random.randint(0,100000)
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        self.rs = np.random.RandomState(self.seed)
        
        if validation is None:
          data = [self.random_sequence(seq_len,T) for _ in range(n_samples)]
        else:
          data = []
          for i in range(n_samples):
            x = self.random_sequence(seq_len,T)
            if x not in validation:
              data.append(x)
          #print(len(data),n_samples)
              
        self.X = np.vstack(data)
        #self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = np.array(hist(self.X[i]))

    def random_sequence(self,n,T):
      partition = self.random_partition(n)
      partition = np.cumsum(partition)
      if len(partition) > T:
        partition = partition[:T-1] + sum(partition[T-1:])
      tokens = self.rs.choice(range(T),size=len(partition),replace=False)
      sequence = np.zeros(n)
      sequence[:partition[0]] = tokens[0]
      for i,j,t in zip(partition[:-1],partition[1:],tokens[1:]):
        sequence[i:j] = t
      self.rs.shuffle(sequence)
      return np.array(sequence)

    def random_partition(self,n):
      if n == 0:
        return []
      if n == 1:
        return [1]
      k = self.rs.randint(1,n)
      return [k] + self.random_partition(n-k)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)
      
class HistogramDatasetSampleBackwardWithBOS(HistogramDatasetSampleBackward):
    has_BOS = True
    
    def __init__(self, seq_len, T, n_samples,seed=None,validation=None):
        super(HistogramDatasetSampleBackwardWithBOS, self).__init__(seq_len, T, n_samples,seed=seed,validation=validation)
        self.X = np.concatenate([np.ones((self.n_samples,1)) * T,self.X],axis=1)
        self.y = np.concatenate([np.ones((self.n_samples,1)) * T,self.y],axis=1)
        
class HistogramDatasetSampleBackwardEmbdWithBOS(HistogramDatasetSampleBackwardEmbd):
    has_BOS = True
    
    def __init__(self, seq_len, T, n_samples,seed=None,validation=None,embedding_dim=100,eps=0.001, mus=None):
        if mus is None:
          self.mus = np.random.normal(loc=0, scale=np.sqrt(1/embedding_dim), size=(T+1, embedding_dim))
        else: 
          self.mus = mus
        super().__init__(seq_len, T, n_samples,seed=seed,
                                                                        validation=validation,embedding_dim=embedding_dim,
                                                                        eps=eps, mus=self.mus)
        
        self.X = np.concatenate([np.ones((self.n_samples,1)) * T,self.X],axis=1)
        self.y = np.concatenate([np.ones((self.n_samples,1)) * T,self.y],axis=1)
        
        self.X = self.X.astype(np.int32)
    

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        pe = torch.arange(0, max_seq_length)
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        e = self.embedding(self.pe)
        return torch.tile(e,(x.shape[0], 1, 1))
      
class DotPMixer(nn.Module): 
  def __init__(self,model_dim,seq_len,attention_input='both'):
    super(DotPMixer, self).__init__()
    self.model_dim = model_dim
    self.seq_len = seq_len
    if attention_input not in ['both', 'only_pos', 'only_sem']:
      raise ValueError()
    self.attention_input = attention_input
    self.Q = nn.Parameter(torch.empty(model_dim,model_dim,device=device))
    self.K = nn.Parameter(torch.empty(model_dim,model_dim,device=device))
    nn.init.kaiming_uniform_(self.Q.T, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.K.T, a=math.sqrt(5))
    
    
  def forward(self,x_sem, x_pos):
    if self.attention_input == 'both':
      x = x_sem + x_pos
    elif self.attention_input == 'only_sem':
      x = x_sem
    elif self.attention_input == 'only_pos':
      x = x_pos
    else:
      raise ValueError()
    Qx = x @ self.Q
    Kx = x @ self.K
    return torch.matmul(Qx,Kx.transpose(-2,-1)) / math.sqrt(self.model_dim)
  
      
class LinearMixer(nn.Module):
  def __init__(self,model_dim,seq_len):
    super(LinearMixer, self).__init__()
    self.model_dim = model_dim
    self.seq_len = seq_len
    self.A = nn.Parameter(torch.empty(seq_len,seq_len,device=device))
    nn.init.kaiming_uniform_(self.A.T, a=math.sqrt(5))

  def forward(self,x_sem,x_pos):
    return self.A.unsqueeze(0)#.repeat(x_sem.shape[0],1,1) unclear if the latter is actually needed of if broadcasting works correctly.

class TransformerSeq2Seq(nn.Module):

  def __init__(self,T,model_dim,p,n_classes,L,attention_input,use_softmax=True,is_embedded=False):
    super(TransformerSeq2Seq, self).__init__()
    
    self.is_embedded = is_embedded

    self.model_dim = model_dim
    self.attention_input = attention_input
    embedding_dim = model_dim
    self.use_softmax = use_softmax

    self.semantic_emb = nn.Embedding(T, embedding_dim)
    self.positional_emb = LearnedPositionalEncoding(embedding_dim,L)

    if not (self.attention_input == 'both' or self.attention_input == 'only_pos'):
      self.positional_emb.requires_grad_(False)
    
    if self.attention_input  == 'linear':
      self.token_mixer = LinearMixer(model_dim,L)
    else:
      self.token_mixer = DotPMixer(model_dim,L,attention_input=attention_input)
    
    #self.norm = nn.LayerNorm(model_dim)
    self.fc1 = nn.Linear(model_dim, p)
    self.activ = nn.ReLU()
    self.fc2 = nn.Linear(p, n_classes)

  def forward(self,x): # B x L
    if self.is_embedded:
      x_sem = x
    else:
      x_sem = self.semantic_emb(x) # B x L x d/2 or B x L x d
      
    if self.attention_input == 'both' or self.attention_input == 'only_pos':
      x_pos = self.positional_emb(x)
      attn_scores = self.token_mixer(x_sem,x_pos) # B x L x L 
    else:
       attn_scores = self.token_mixer(x_sem,None) # B x L x L 
    
    if self.use_softmax:
      attn_probs = torch.softmax(attn_scores,dim=-1)
    else:
      attn_probs = attn_scores
    self.attn_probs = attn_probs
    
    a = torch.matmul(attn_probs,x_sem)
    self.a = a
    if self.attention_input == 'both' and not self.is_embedded:
      x = a + x_sem + x_pos
    elif self.attention_input == 'only_sem':
      x = a + x_sem
    elif self.attention_input == 'only_pos':
      x = a + x_sem
    else:
      x = a + x_sem
    self.x = x
    self.b = self.fc1(x)
    x = self.fc2(self.activ(self.b))
    return x
  
  def get_output(self,a,x_sem):
    x = a + x_sem
    x = self.fc2(self.activ(self.fc1(x)))
    return x
  
class Transformer2Layer(nn.Module):
  
  def __init__(self,T,model_dim,p,n_classes,L,attention_input,use_softmax=True,is_embedded=False):
    super(Transformer2Layer, self).__init__()
    
    self.layer1 = TransformerSeq2Seq(T,model_dim,p,model_dim,L,attention_input,use_softmax,is_embedded)
    self.layer2 = TransformerSeq2Seq(T,model_dim,p,n_classes,L,attention_input,use_softmax,is_embedded)
    
  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(transformer, train_dataset, val_dataset, n_classes, has_BOS,lr = 0.001,n_epochs = 100, optimizer=None,batch_size=32):
  wandb.log({'params_count':count_parameters(model=model)})
  
  val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
  
  criterion = nn.CrossEntropyLoss()
  if optimizer == 'Adam':
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
  elif optimizer == 'AdamW':
    optimizer = optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
  else:
    raise ValueError(f'optimizer {optimizer} not supported')

  train_losses = []
  val_losses = []
  val_acc = []

  for epoch in range(n_epochs):
          train_dataloader = DataLoader(train_dataset(), batch_size=batch_size, shuffle=True)
          epoch_loss = 0.0
          train_acc= 0.0

          for X, y in train_dataloader:
              X = X.to(device)
              optimizer.zero_grad()
              
              if has_BOS:
                y = y.to(device)[...,1:]
                output = transformer(X)[...,1:,:]
              else:
                y = y.to(device)
                output = transformer(X)
                
              loss = criterion(output.reshape(-1, n_classes), y.reshape(-1))

              epoch_loss += loss.item()
              train_acc += torch.mean((output.argmax(axis=-1).reshape(-1)==y.reshape(-1)).float()).item()

              loss.backward()
              optimizer.step()

          epoch_loss /= len(train_dataloader)
          train_acc /= len(train_dataloader)
          train_losses.append(epoch_loss)

          # Evaluate on the test set every epoch
          with torch.no_grad():
              val_loss = 0.0
              acc = 0.0
              for X, y in val_dataloader:
                  X = X.to(device)
                  
                  if has_BOS:
                        y = y.to(device)[...,1:]
                        output = transformer(X)[...,1:,:]
                  else:
                        y = y.to(device)
                        output = transformer(X)
                  
                  pred = output.argmax(axis=-1)
                  loss = criterion(output.reshape(-1,n_classes), y.reshape(-1))
                  acc += torch.mean((pred.reshape(-1)==y.reshape(-1)).float()).item()
                  val_loss+=loss.item()
              val_loss /= len(val_dataloader)
              acc /= len(val_dataloader)

              wandb.log({'train_loss': epoch_loss, 'val_loss': val_loss, 'val_acc': acc*100, 'epoch': epoch})
              val_losses.append(val_loss)
              val_acc.append(acc)

          if epoch % 10 == 0:
            print(f'[Epoch {epoch:02}] Train loss = {epoch_loss:.5f} :: {train_acc:.5f} :: Val loss {val_loss:.5f} :: Val accuracy {acc*100:.2f}')
  return transformer, train_losses, val_losses, val_acc


def predict_sequence(x,transformer):
  x_ = torch.tensor(x,dtype=torch.long,device=device).unsqueeze(0)
  pred = transformer(x_).argmax(axis=-1).detach().cpu().numpy()
  y = hist(torch.Tensor(x))
  return list(pred), np.all(y == pred)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transformer Memory Experiment')
    parser.add_argument('--project_name', type=str, default='memorizing_counts', help='Project name')
    parser.add_argument('--seq_len', type=int, default=11, help='Length of the sequence')
    parser.add_argument('--T', type=int, default=200, help='Maximum value')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--p', type=int, default=128, help='MLP dimension')
    parser.add_argument('--model_dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--attention_input', type=str, default='both', choices=['both', 'only_pos', 'only_sem', 'linear'], help='Attention input type')
    parser.add_argument('--dataset_type', type=str, default='forward', choices=['forward', 'backward','backward_BOS'], help='Dataset type')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_softmax', action='store_true', help='Do not use softmax')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--include_validation', action='store_true', help='Include validation samples in the training set')
    parser.add_argument('--offline_learning', action='store_true', help='Offline learning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_type', type=str, default='1layer', choices=['1layer','2layer'], help='Model type')
    
    
    args = parser.parse_args()

    config = vars(args)

    if args.run_name: 
      run = wandb.init(
          project=args.project_name,
          name=args.run_name,
          config=config
      )
    else:
      run = wandb.init(
          project=args.project_name,
          config=config
      )  

    dataset_cls = {
                'backward': HistogramDatasetSampleBackwardEmbd,
                'backward_BOS': HistogramDatasetSampleBackwardEmbdWithBOS
                }[config['dataset_type']]
    # Create the dataset
    val_size = int(0.3 * config['num_samples'])
    val_dataset = dataset_cls(config['seq_len'], config['T'], val_size,embedding_dim=config['model_dim'],seed = 12)
    validation = ExclusionDataset(val_dataset.X[:,int(val_dataset.has_BOS):].copy())
    if val_dataset.has_BOS:
      bos = val_dataset.mus[args.T]
      np.save('BOS',bos)
      artifact = wandb.Artifact(f'{wandb.run.name}-BOS', type='model')
      artifact.add_file('BOS.npy')
      run.log_artifact(artifact)
    if not config['offline_learning']:
      train_dataset = lambda: dataset_cls(config['seq_len'], config['T'], config['num_samples'],embedding_dim=config['model_dim'],validation=validation if not config['include_validation'] else None, 
                                          mus=val_dataset.mus)
    else:
      # use the same dataset in every epoch
      train_dataset = dataset_cls(config['seq_len'], config['T'], config['num_samples'],embedding_dim=config['model_dim'], validation=validation if not config['include_validation'] else None, mus=val_dataset.mus)

    model_cls = {'1layer': TransformerSeq2Seq,
                '2layer': Transformer2Layer}[config['model_type']]
    model = model_cls(T=config['T']+int(val_dataset.has_BOS),
                                            model_dim=config['model_dim'],
                                            p=config['p'],n_classes=config['seq_len']+1,
                                            L=config['seq_len'],
                                            is_embedded=True,
                                            
                                            attention_input=config['attention_input'],use_softmax=not config['no_softmax']).to(device)

    
    
    model, train_losses, val_losses, val_acc = train(model, train_dataset, val_dataset,config['seq_len']+1,val_dataset.has_BOS,lr=config['lr'],n_epochs=config['n_epochs'],optimizer=config['optimizer'],batch_size=config['batch_size'])

    # save the model as a wandb artifact
    # get the name of the current wandb run
    run_name = wandb.run.name
    torch.save(model.state_dict(), 'transformer.pt')
    artifact = wandb.Artifact(f'{wandb.run.name}-model', type='model')
    artifact.add_file('transformer.pt')
    run.log_artifact(artifact)
    run.finish()





