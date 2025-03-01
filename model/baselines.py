import torch
import torch.nn as  nn
from model.Seqmodels import *
from pyhealth.medcode import InnerMap
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence
from torch.nn.utils.rnn import PackedSequence, pad_sequence

# Adapted from https://github.com/The-Real-JerryChen/TRANS

def multihot_to_embedding(x, embedding, embedding_key):
    # x: (visit, one_hot)
    out = []
    for visit in x:
        codes = torch.where(visit == 1)[0]
        codes = codes.long().to(embedding[embedding_key].weight.device)
        embedding_vec = embedding[embedding_key](codes)
        embedding_vec = torch.sum(embedding_vec, dim=0)
        out.append(embedding_vec)
    out = torch.stack(out)
    return out

class CNN(nn.Module):
    def __init__(
        self,
        tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        num_layers = 4,
        dropout = 0.5,
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
    ):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = tokenizers
        self.embeddings = nn.ModuleDict()
        self.feature_keys = tokenizers.keys()
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input

        for feature_key in self.feature_keys:
            self.add_feature_layer(feature_key)

        self.network = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.network[feature_key] = nn.Sequential(
                nn.Dropout(dropout),
                *([
                    nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(embedding_dim),
                    nn.ReLU(),
                    nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(embedding_dim),
                ] * (num_layers)),
            )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}

        patient_emb = []
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x_ = [(torch.cat([x_i]*3, dim=0) if len(x_i) == 1 else x_i) for x_i in x_]
                    x = [x_i.to(self.device) for x_i in x_]
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
            
            if isinstance(x, list):
                x = [self.network[feature_key](x_i.permute(1, 0).unsqueeze(0)) for x_i in x]
                x = [self.avgpool(x_i).squeeze() for x_i in x]
                x = torch.stack(x)
            else:
                x = self.network[feature_key](x.permute(0, 2, 1))
                x = self.avgpool(x).squeeze()
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)

        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits

class FFN(nn.Module):
    def __init__(
        self,
        tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        num_layers = 4,
        dropout = 0.5,
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
    ):
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = tokenizers
        self.embeddings = nn.ModuleDict()
        self.feature_keys = tokenizers.keys()
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input

        for feature_key in self.feature_keys:
            self.add_feature_layer(feature_key)

        self.network = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.network[feature_key] = nn.Sequential(
                nn.Dropout(dropout),
                *([
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(embedding_dim),
                ] * (num_layers)),
            )
        
        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}

        patient_emb = []
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x_ = [x_i.mean(dim=0) for x_i in x_]
                    x = torch.stack(x_).to(self.device)
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
            
            x = self.network[feature_key](x)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)

        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits
    

class Dipole(nn.Module):
    def __init__(
        self,
        tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        dropout = 0.5,
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
    ):
        super(Dipole, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = tokenizers
        self.embeddings = nn.ModuleDict()
        self.feature_keys = tokenizers.keys()
        self.device = device
        self.dropout = dropout
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input

        for feature_key in self.feature_keys:
            self.add_feature_layer(feature_key)

        self.network = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.network[feature_key] = DipoleLayer(feature_size=embedding_dim, dropout = dropout)
        
        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}

        patient_emb = []
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x = pad_sequence(x_, batch_first=True).to(self.device)
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
            
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            x = self.network[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)

        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits

class Transformer(nn.Module):
    def __init__(
        self,
        tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        dropout = 0.5, 
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
    ):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = tokenizers
        self.embeddings = nn.ModuleDict()
        self.feature_keys = tokenizers.keys()
        self.device = device
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input

        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                feature_size=embedding_dim, dropout = dropout,num_layers=2
            )
        
        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}

        patient_emb = []
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x = pad_sequence(x_, batch_first=True).to(self.device)
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)

        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits
    
class RETAIN(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
        embedding_dim: int = 128, dropout = 0.5,
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
        ):
        super(RETAIN, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input
        self.feat_tokenizers = Tokenizers
        self.feature_keys = Tokenizers.keys()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout = dropout)

        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)


    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}

        patient_emb = []
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x = pad_sequence(x_, batch_first=True).to(self.device)
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                x = torch.sum(x, dim=2)
            
            mask = torch.sum(x, dim=2) != 0
            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)
        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)
        logits = self.fc(patient_emb)
        return logits
    
class StageNet(nn.Module):
    def __init__(self, Tokenizers, output_size, device, embedding_dim: int = 128,
        chunk_size: int = 128,
        levels: int = 3,
        static_input=False, static_input_dim=None, 
        use_all_seq=False,
        multihot_input=True
    ):
        super(StageNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq
        self.multihot_input = multihot_input
        self.device = device
        
        self.feature_keys = Tokenizers.keys()

        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()

        self.stagenet = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
            )

        if static_input:
            self.fc = nn.Linear(len(self.feature_keys) * self.chunk_size * self.levels + static_input_dim, output_size)
        else:
            self.fc = nn.Linear(
                len(self.feature_keys) * self.chunk_size * self.levels, output_size
            )

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                # padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):
        if self.static_input:
            batchdata, static_input = batchdata
        batchdata = {'cond_hist': batchdata}
        
        patient_emb = []
        distance = []
        mask_dict = {}
        for feature_key in self.feature_keys:
            x = batchdata[feature_key]
            if self.multihot_input:
                if isinstance(x, PackedSequence):
                    x_ = unpack_sequence(x)
                    x_ = [multihot_to_embedding(x_i, self.embeddings, feature_key) for x_i in x_]
                    x = pad_sequence(x_, batch_first=True).to(self.device)
                else:
                    x = multihot_to_embedding(x, self.embeddings, feature_key)
                    x = torch.stack(x).to(self.device)
            else:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    x,
                )
    
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            mask_dict[feature_key] = mask
            time = None
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        if self.static_input:
            patient_emb = torch.cat([patient_emb, static_input], dim=1)
        logits = self.fc(patient_emb)
        return logits