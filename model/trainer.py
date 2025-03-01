import torch
from torchvision import transforms
from torch.nn.utils.rnn import PackedSequence, unpack_sequence, pack_sequence

class AETrainer:
    def __init__(self, hparams, model, optimizer, criterion):
        self.hparams = hparams
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def update(self, mnb):
        x = mnb
        if self.hparams['encoder_type'] == 'resnet':
            x_ = transforms.Resize((256, 256))(x)
        else:
            x_ = x
        x_hat = self.model(x_)
        loss = self.criterion(x_hat, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    
    def evaluate(self, mnb):
        x = mnb
        # x = x.to(self.hparams['device'])
        x_hat = self.model(x)
        loss = self.criterion(x_hat, x)
        return {'loss': loss.item()}

class LSTMTrainer:
    def __init__(self, hparams, model, optimizer, criterion):
        self.hparams = hparams
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def update(self, mnb):
        if len(mnb) == 2:
            x, y = mnb
            y_hat = self.model(x)
        else:
            x, y, c = mnb
            y_hat = self.model((x, c))
        
        if isinstance(y_hat, PackedSequence):
            assert isinstance(y, PackedSequence)
            y_hat = unpack_sequence(y_hat)
            y = unpack_sequence(y)
            losses = [self.criterion(y_hat_i, y_i) for y_hat_i, y_i in zip(y_hat, y)]
            loss = torch.stack(losses).mean()
        else:
            loss = self.criterion(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    
    def evaluate(self, mnb):
        if len(mnb) == 2:
            x, y = mnb
            y_hat = self.model(x)
        else:
            x, y, c = mnb
            y_hat = self.model((x, c))
        
        if isinstance(y_hat, PackedSequence):
            assert isinstance(y, PackedSequence)
            y_hat = unpack_sequence(y_hat)
            y = unpack_sequence(y)
            losses = [self.criterion(y_hat_i, y_i) for y_hat_i, y_i in zip(y_hat, y)]
            loss = torch.stack(losses).mean()
            y_hat = torch.cat(y_hat, dim=0)
            y = torch.cat(y, dim=0)
        else:
            loss = self.criterion(y_hat, y)
            
        out_loss = {'loss': loss.item()}
        return out_loss, y_hat, y