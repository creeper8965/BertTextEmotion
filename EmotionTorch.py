import torch
from torch import nn, Tensor, optim
F = nn.functional
from bertTorch import Bert
import pandas as pd
from os.path import expanduser
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm, trange
import numpy as np

#Params
#Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
DEVICE = torch.device('mps')
STEPS = 100
Batch_Size = 32
LR = 1e-4
Max_len = 320
Num_Classes = 6
ids_to_names = {0:'sadness',1:'joy',2:'love',3:'anger',4:'fear',5:'suprise'}

Tokenizer = SentencePieceProcessor() ##Tokenizer.Encode(input='Hi There ! #2'.lower(),out_type=int,add_eos=True)
Tokenizer.Load(model_file='en.wiki.5000.model')

def prepare_toks(x, max_len):
    if x.__len__() < max_len:
        pad_len = max_len - x.__len__()
        pad_toks = [0]*pad_len
        final_token = x + pad_toks
        attn = [1]*x.__len__()+pad_toks
    elif x.__len__() >= max_len:
        final_token = x[:max_len]
        attn = [1]*max_len

    return (final_token, attn)


class BertCLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = Bert(hidden_size=320,intermediate_size=512,max_position_embeddings=320,num_attention_heads=4,num_hidden_layers=4,vocab_size=Tokenizer.vocab_size(),attention_probs_dropout_prob=0.1,hidden_dropout_prob=0.1 )
        self.fc = nn.Sequential(nn.Conv1d(320,240,1,1),nn.ReLU(),nn.Conv1d(240,120,1,1),nn.ReLU(),nn.Conv1d(120,60,1,1),nn.ReLU(),nn.Conv1d(60,30,1,1),nn.ReLU(),nn.Conv1d(30,10,1,1),nn.ReLU(),nn.Flatten(1), nn.Linear(3200, 100),nn.ReLU(),nn.Linear(100,6))

    def __call__(self,ids,attn):
        x = self.bert(ids,attn)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    dataframe = pd.read_csv(expanduser('~/Desktop/EmptyIdeas/TwitterEmotionDataset2.csv'))
    tokensList = []
    attnList = []

    print('Processing data is fast but transfering to Tensors is slow.')
    for row in tqdm(dataframe['text']):
        t,a = prepare_toks(Tokenizer.Encode(input=row.lower(),out_type=int), Max_len)
        tokensList.append(t)
        attnList.append(a)

    XtensorIds = Tensor(tokensList).long().to(DEVICE)
    #(416809, 320)
    XtensorAttn = Tensor(attnList).long().to(DEVICE)
    #(416809, 320)

    Y = np.array([dataframe['label'][i] for i in range(len(dataframe['label']))])
    Y = np.eye(Num_Classes)[Y]
    Ytensor = Tensor(Y).to(DEVICE)
    #(416809,6)
    print('Data processing finished.')

    model = BertCLS()
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(),LR)

    def train_step(XtensorIds, XtensorAttn,Ytensor):
        model.train()
        opt.zero_grad()
        samples = torch.randint(size=(Batch_Size,), high=XtensorIds.shape[0])
        XtensorIds, XtensorAttn,Yt = XtensorIds[samples], XtensorAttn[samples], Ytensor[samples]  # we shard the data on axis 0
        logits = model(XtensorIds, XtensorAttn)
        loss = F.cross_entropy(logits, Yt)
        loss.backward()
        opt.step()
        return loss

    test_acc = float('nan')
    for i in (t:=trange(STEPS)):
        loss = train_step(XtensorIds, XtensorAttn,Ytensor)
        t.set_description(f"loss: {loss.item():6.2f}")

    StDict = model.state_dict()
    torch.save(StDict, 'EmotionalDamage.pt')
