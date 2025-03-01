import os
import torch
import numpy as np
from argparse import ArgumentParser
from dataset import InfiniteDataLoader, FastDataLoader, MarketScanDataset
from model import LSTM
from model.baselines import Transformer, StageNet, RETAIN, Dipole, FFN, CNN
from utils import Metrics
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from model import LSTMTrainer
from tqdm import tqdm
import random
import pickle as pkl
import json
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--checkpoint_freq', type=int, default=200)
parser.add_argument('--feature_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--label_type', type=str, default='raw')
parser.add_argument('--model_dir', type=str, default='saved_models/nextvisit')
parser.add_argument('--score_dir', type=str, default='scores/nextvisit')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--use_climate', action='store_true')
parser.add_argument('--use_all_seq', action='store_true')
parser.add_argument('--window_size', type=int, default=0)
args = parser.parse_args()

num_epochs = args.num_epochs
lr = args.lr
checkpoint_freq = args.checkpoint_freq
feature_dim = args.feature_dim
batch_size = args.batch_size
random_state = args.random_state
seed = args.seed
device = args.device
num_workers = args.num_workers
label_type = args.label_type
model_dir = args.model_dir
score_dir = args.score_dir
use_climate = args.use_climate
use_all_seq = args.use_all_seq
test_only = args.test_only if num_epochs > 0 else True
window_size = args.window_size if args.window_size > 0 else None
suffix = '_' + args.suffix if args.suffix != '' else ''
if test_only:
    num_epochs = 0
os.makedirs(model_dir, exist_ok=True)
os.makedirs(score_dir, exist_ok=True)

# parameter initialization
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if device == 'cuda':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
report_freq = checkpoint_freq // 4 if checkpoint_freq > 4 else 1
args.device = device
args.report_freq = report_freq
print(args)

if use_climate:
    market_scan_Y_df = pd.read_csv(f'data/processed/marketscan/pivot/market_scan_Y.csv', index_col=[0, 1]).dropna(how='any', axis=1)
    combined_X_df = pd.read_csv(f'data/processed/marketscan/pivot/combined_X.csv', index_col=[0, 1]).loc[market_scan_Y_df.index]
    all_env_data = combined_X_df.reset_index().rename(columns={'MSA': 'GEOID'})
    all_env_data['GEOID'] = all_env_data['GEOID'].astype(str)
else:
    all_env_data = None

train_dataset = MarketScanDataset(root_dir='data/processed/marketscan', split='train', climate_data=all_env_data, use_all_seq=use_all_seq, window_size=window_size, label_type=label_type)
val_dataset = MarketScanDataset(root_dir='data/processed/marketscan', split='val', climate_data=all_env_data, use_all_seq=use_all_seq, window_size=window_size, label_type=label_type)
test_dataset = MarketScanDataset(root_dir='data/processed/marketscan', split='test', climate_data=all_env_data, use_all_seq=use_all_seq, window_size=window_size, label_type=label_type, ret_date=True)
train_dataloader = iter(InfiniteDataLoader(train_dataset, weights=None, batch_size=batch_size, num_workers=num_workers, collate_fn=train_dataset.get_collate_fn()))
val_dataloader = FastDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=val_dataset.get_collate_fn())
test_dataloader = FastDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=test_dataset.get_collate_fn())

tokenizer = train_dataset.get_tokenizer()

vocab_size = train_dataset.code_vocabulary_size
static_input_dim = all_env_data.shape[1] - 2 if use_climate else None

if args.model == 'lstm':
    model = LSTM(input_dim=vocab_size, latent_dim=feature_dim, num_layers=1, 
                num_classes=vocab_size, static_input=use_climate, static_input_dim=static_input_dim,
                use_all_seq=use_all_seq)
elif args.model == 'transformer':
    model = Transformer(tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                        static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
elif args.model == 'stagenet':
    model = StageNet(Tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                     static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
elif args.model == 'retain':
    model = RETAIN(Tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                   static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
elif args.model == 'dipole':
    model = Dipole(tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                   static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
elif args.model == 'ffn':
    model = FFN(tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
elif args.model == 'cnn':
    model = CNN(tokenizers=tokenizer, output_size=vocab_size, device=device, embedding_dim=feature_dim,
                static_input=use_climate, static_input_dim=static_input_dim, use_all_seq=use_all_seq)
else:
    raise NotImplementedError(f'Model {args.model} not implemented!')

model = model.to(device)
critertion = BCEWithLogitsLoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)
metrics = Metrics()

trainer = LSTMTrainer({}, model, optimizer, critertion)

err_seq = 0
all_losses = {}
running_losses = {}
best_loss = float('inf')
best_auroc = 0
best_epoch = -1
for epoch in tqdm(range(num_epochs)):
    model.train()
    mnb = next(train_dataloader)
    mnb = [x.to(device) for x in mnb] if isinstance(mnb, tuple) else mnb.to(device)
    try:
        loss = trainer.update(mnb)
    except RuntimeError as e:
        print(f'Error: {e}')
        os.makedirs(f'error_batches/{args.model}_{label_type}{suffix}', exist_ok=True)
        with open(f'error_batches/{args.model}_{label_type}{suffix}/error_batch_{err_seq}.pkl', 'wb') as f:
            mnb = [x.to('cpu') for x in mnb] if isinstance(mnb, list) else mnb.to('cpu')
            pkl.dump(mnb, f)
        err_seq += 1
        print(f'Skipping batch...')
        continue
    torch.cuda.empty_cache()
    for key, value in loss.items():
        if key not in running_losses:
            running_losses[key] = []
        running_losses[key].append(value)

    if ((epoch + 1) % report_freq == 0) or (epoch == num_epochs - 1):
        print(f'Epoch {epoch} - Training Loss:')
        for key, value in running_losses.items():
            if key not in all_losses:
                all_losses[key] = []
            all_losses[key].extend(value)
            mean_loss = np.mean(value)
            print(f'\t{key}: {mean_loss:.4f}')
            running_losses[key] = []

    if ((epoch + 1) % checkpoint_freq == 0) or (epoch == num_epochs - 1):
        model.eval()
        val_losses = {}
        y_true, y_pred = [], []
        with torch.no_grad():
            for val_mnb in tqdm(val_dataloader):
                val_mnb = [x.to(device) for x in val_mnb] if isinstance(val_mnb, tuple) else val_mnb.to(device)
                try:
                    val_loss, y_pred_, y_true_ = trainer.evaluate(val_mnb)
                except RuntimeError as e:
                    print(f'Error: {e}')
                    os.makedirs(f'error_batches/{args.model}_{label_type}{suffix}', exist_ok=True)
                    with open(f'error_batches/{args.model}_{label_type}{suffix}/error_batch_{err_seq}.pkl', 'wb') as f:
                        mnb = [x.to('cpu') for x in mnb] if isinstance(mnb, list) else mnb.to('cpu')
                        pkl.dump(val_mnb, f)
                    err_seq += 1
                    print(f'Skipping batch...')
                    continue
                y_true.append(y_true_.detach().cpu())
                y_pred.append(y_pred_.detach().cpu())
                for key, value in val_loss.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value)
                break
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        val_metrics = metrics(y_true, y_pred)
        print(f'Epoch {epoch} - Val Losses:')
        for key, value in val_losses.items():
            mean_loss = np.mean(value)
            print(f'\t{key}: {mean_loss:.4f}')
        print(f'Epoch {epoch} - Val Metrics:')
        for key, value in val_metrics.items():
            if key == 'avg' or key.startswith('recall@'):
                if key != 'avg':
                    print(f'\t{key}: {value}')
                continue
            mean_metric = np.mean(value)
            print(f'\t{key}: {mean_metric:.4f} [{", ".join([f"{v:.4f}" for v in value])}]')
        
        # model selection
        val_loss = np.mean(val_losses['loss'])
        val_auroc = val_metrics['avg']['auroc']
        # if val_loss < best_loss:
        if val_auroc > best_auroc:
            best_loss = val_loss
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save({'state_dict':model.state_dict(),
                        'best_epoch':best_epoch,
                        'best_auroc':best_auroc,
                        'best_loss':best_loss,
                        'args': args.__dict__,
                        'feature_dim': feature_dim,
                        'random_state': random_state,
                        'batch_size': batch_size,
                        'seed': seed}, 
                        os.path.join(model_dir, f'best_model_{args.model}_{label_type}{suffix}.pth'))
            with open(os.path.join(score_dir, f'best_val_metrics_{args.model}_{label_type}{suffix}.json'), 'w') as f:
                json.dump(val_metrics, f, indent=4)
            print(f'Epoch {epoch} - Best model saved with loss: {best_loss:.4f}')

if not test_only:
    print(f'Best model found at epoch {best_epoch} with loss: {best_loss:.4f}')
    print('Training complete!')

model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_{args.model}_{label_type}{suffix}.pth'))['state_dict'])
# model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_cli.pth'), map_location=device)['state_dict'])
model.eval()
test_losses = {}
y_pred, y_true, dates = [], [], []
with torch.no_grad():
    for test_mnb in tqdm(test_dataloader):
        *test_mnb, dates_ = test_mnb
        test_mnb = [x.to(device) for x in test_mnb] if isinstance(test_mnb, list) else test_mnb.to(device)
        test_loss, y_pred_, y_true_ = trainer.evaluate(test_mnb)
        y_true.append(y_true_.detach().cpu())
        y_pred.append(y_pred_.detach().cpu())
        if isinstance(dates_, list):
            dates_ = np.concatenate(dates_, axis=0)
        dates.append(dates_)
        for key, value in test_loss.items():
            if key not in test_losses:
                test_losses[key] = []
            test_losses[key].append(value)
y_true = torch.cat(y_true, dim=0)
y_pred = torch.cat(y_pred, dim=0)
dates = np.concatenate(dates, axis=0)
test_metrics = metrics(y_true, y_pred)
print(f'Test Losses:')
for key, value in test_loss.items():
    mean_loss = np.mean(value)
    print(f'\t{key}: {mean_loss:.4f}')
print(f'Test Metrics:')
for key, value in test_metrics.items():
    if key == 'avg' or key.startswith('recall@'):
        if key != 'avg':
            print(f'\t{key}: {value}')
        continue
    mean_metric = np.mean(value)
    print(f'\t{key}: {mean_metric:.4f} [{", ".join([f"{v:.4f}" for v in value])}]')
with open(os.path.join(score_dir, f'test_metrics_{args.model}_{label_type}{suffix}.json'), 'w') as f:
    json.dump(test_metrics, f, indent=4)

test_metrics_peryear = metrics.get_peryear_metrics(y_true, y_pred, dates, datefmt='%Y-%m-%d')
with open(os.path.join(score_dir, f'test_metrics_peryear_{args.model}_{label_type}{suffix}.json'), 'w') as f:
    json.dump(test_metrics_peryear, f, indent=4)

os.makedirs(os.path.join(model_dir, 'test_predictions'), exist_ok=True)
with open(os.path.join(model_dir, 'test_predictions', f'test_predictions_{args.model}_{label_type}{suffix}.pkl'), 'wb') as f:
    pkl.dump({
        'y_true': y_true.numpy(),
        'y_pred': y_pred.numpy(),
        'dates': dates,
        'codes': test_dataset.tokenizer.convert_indices_to_tokens([i for i in range(vocab_size)])
    }, f)

if not test_only:
    with open(os.path.join(score_dir, f'losses_{args.model}_{label_type}{suffix}.pkl'), 'wb') as f:
        pkl.dump({
            'train': {
                'mean':{key: np.mean(value) for key, value in all_losses.items()},
                'all': all_losses
            },
            'test': {key: np.mean(value) for key, value in test_losses.items()}
        }, f)

print('Testing complete!')
