import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import pytorch_lightning as pl
from utils.utils import load_obj
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from preprocess.DataLoader import CPCDataLoader

class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(params['vocab_size'], params['hidden_size'])
        self.age_embeddings = nn.Embedding(params['age_vocab_size'], params['hidden_size'])
        self.LayerNorm = Bert.modeling.BertLayerNorm(params['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(params['hidden_dropout_prob'])

    def forward(self, word_ids, age_ids):
        word_embed = self.word_embeddings(word_ids)
        age_embed = self.age_embeddings(age_ids)
        embeddings = age_embed + word_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CPC(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # set up parameters
        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())
        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.timestep = self.params['time_step']
        self.embedding = Embedding(self.params)
        self.gru = nn.GRU(input_size=self.params['hidden_size'], hidden_size=self.params['hidden_size'],
                           num_layers=self.params['num_layer'], bidirectional=False, batch_first=True, dropout=self.params['hidden_dropout_prob'])
        self.Wk = nn.ModuleList([nn.Linear(self.params['hidden_size'], self.params['hidden_size']) for _ in range(self.timestep)])

        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size):
        return torch.zeros(self.params['num_layer'], batch_size, self.params['hidden_size'], device=self.device)

    def shared_step(self, batch):
        z = self.embedding(batch['code'], batch['age'])
        size = z.size()[0]
        t_samples = torch.randint(torch.max(batch['len'].view(-1)) - self.timestep, size=(1,)).long()

        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep, size, self.params['hidden_size']), device=self.device).float()
        pad_samples = torch.empty(self.timestep, size, device=self.device) # save padding indicator, cancel loss for paddings

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(size, self.params['hidden_size'])
            pad_samples[i - 1] = batch['att_mask'][:, t_samples + i].view(size)

        forward_seq = z[:, :t_samples + 1, :]
        hidden = self.init_hidden(size)
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:, t_samples, :].view(size, self.params['hidden_size'])
        pred = torch.empty((self.timestep, size, self.params['hidden_size']), device=self.device).float()

        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512

        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8 (batch * batch)
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, size, device=self.device)) * pad_samples[i])  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)) * pad_samples[i])  # nce is a tensor

        nce /= -1. * size * self.timestep
        accuracy = 1. * correct.item() / size

        return accuracy, nce

    def training_step(self, batch, batch_idx):
        acc, loss = self.shared_step(batch)
        self.log('loss', loss, on_epoch=True, sync_dist=True)
        self.log('accuracy', acc, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params['optimiser_params']['lr'])
        scheduler = CosineAnnealingLR(
            optimizer,
            **self.params['scheduler']
        )
        return [optimizer], [scheduler]


class CPCDL(pl.LightningDataModule):
    def __init__(self, data_path, params):
        super().__init__()
        self.data_path = data_path
        self.params = params

    def setup(self, stage):
        data = pd.read_parquet(self.data_path)
        self.train = data.reset_index(drop=True)

    def train_dataloader(self):
        train = CPCDataLoader(params=self.params, batch_size=self.params['batch_size'], data=self.train, shuffle=True)
        return train