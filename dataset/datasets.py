import torch
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json
import pickle as pkl
from pyhealth.tokenizer import Tokenizer

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

class GMapDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=default_transform, random_state=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.random_state = random_state
        self.meta_data = pd.read_csv(os.path.join(root_dir, 'loc_meta.csv'))
        self.image_path = os.path.join(root_dir, 'images')

        if split != 'all':
            train_idx, val_test_idx = train_test_split(range(len(self.meta_data)), test_size=0.2, random_state=random_state)
            val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=random_state)
            if split == 'train':
                self.meta_data = self.meta_data.iloc[train_idx]
            elif split == 'val':
                self.meta_data = self.meta_data.iloc[val_idx]
            elif split == 'test':
                self.meta_data = self.meta_data.iloc[test_idx]

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        item = self.meta_data.iloc[idx]
        img_name = os.path.join(self.image_path, item['filename'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image
    
ohio_msa = ['48540', '26580', '17140', '17460', '10420', '15940', '18140', '19380', '44220', '30620', '31900', '48260', 
            '45780', '49660', '11900', '38580', '18740', '27160', '49300', '23380', '19580', '39020', '48940', '41780', 
            '11780', '35420', '35940', '13340', '17060', '49780', '15740', '32020', '34540', '47920', '46500', '24820', 
            '43380', '22300', '45660', '46780', '16380', '47540', '15340', '11740', '31930', '38840', '41400']

class MarketScanDataset(Dataset):
    src_dir = 'data/processed/marketscan/pat_enrol_clean'
    def __init__(self, root_dir, split='train', transform=default_transform, random_state=42, 
                 climate_data=None, use_all_seq=False, window_size=None, label_type='raw',
                 ret_date=False):
        super(MarketScanDataset).__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.random_state = random_state
        self.climate_data = climate_data
        self.use_all_seq = use_all_seq
        self.window_size = window_size
        self.label_type = label_type
        self.ret_date = ret_date

        # self.meta_data = pd.read_csv(os.path.join(self.src_dir, 'meta_data.csv'), dtype={'enrol_id': str, 'date': str})
        # patient_src = pd.read_csv(os.path.join(self.src_dir, 'patient_src.csv'), dtype={'enrol_id': str, 'MSA': str, 'src': str})
        # patient_src = patient_src[patient_src['MSA'].isin(ohio_msa)]
        # self.meta_data = pd.merge(self.meta_data, patient_src, on='enrol_id')
        # self.meta_data['date'] = pd.to_datetime(self.meta_data['date'])
        # self.meta_data = self.meta_data.sort_values(by=['enrol_id', 'date'])

        # # with open(os.path.join(self.src_dir, 'code_vocabulary.json')) as f:
        # #     self.code_vocabulary = json.load(f)
        # self.code_vocabulary = pd.read_csv(os.path.join(self.src_dir, '../code_count/code_count_pivot.csv'), dtype={'code': str})
        # self.code_vocabulary = self.code_vocabulary['code'].to_list()
        # self.code_vocabulary_size = len(self.code_vocabulary)
        # self.code_mapping = {code: i for i, code in enumerate(sorted(self.code_vocabulary))}
        
        self.meta_data = pd.read_csv(os.path.join(root_dir, 'meta_data_sub.csv'), dtype={'enrol_id': str, 'date': str, 'MSA':str})
        
        with open(os.path.join(root_dir, 'code_vocabulary.json')) as f:
            self.code_vocabulary = json.load(f)

        if self.label_type == 'raw':
            self.code_vocabulary = [code for code in self.code_vocabulary if code[0].isalpha()]
        elif self.label_type == 'l2':
            self.code_vocabulary = [code for code in self.code_vocabulary if code[0].isalpha()]
            with open('data/raw/icd10/icd10l3.json', 'r') as f:
                icd10l3 = json.load(f)
            self.icd_mapping = {k: v['parent'][1] for k,v in icd10l3.items()}
            self.code_vocabulary = list(set([self.icd_mapping[code] for code in self.code_vocabulary if code in self.icd_mapping]))
        elif self.label_type == 'l1':
            self.code_vocabulary = [code for code in self.code_vocabulary if code[0].isalpha()]
            with open('data/raw/icd10/icd10l3.json', 'r') as f:
                icd10l3 = json.load(f)
            self.icd_mapping = {k: v['parent'][0] for k,v in icd10l3.items()}
            self.code_vocabulary = list(set([self.icd_mapping[code] for code in self.code_vocabulary if code in self.icd_mapping]))
        else:
            raise NotImplementedError(f'Unknown label type {self.label_type}')
        
        self.tokenizer = Tokenizer(tokens=self.code_vocabulary)
        self.code_vocabulary_size = len(self.code_vocabulary)
        # self.code_mapping = {code: i for i, code in enumerate(sorted(self.code_vocabulary))}

        # print('Loading enrolls...')
        # self.load_enrolls(self.meta_data)
        # self.enrolls = {k:v for k,v in self.enrolls.items() if len(v) > 1}
        # self.meta_data = self.meta_data[self.meta_data['enrol_id'].isin(self.enrolls.keys())]
        # with open(os.path.join(root_dir, 'enrolls.pkl'), 'wb') as f:
        #     pkl.dump(self.enrolls, f)
        # self.meta_data.to_csv(os.path.join(root_dir, 'meta_data.csv'), index=False)

        # patient_set = set(self.meta_data['enrol_id'])
        # patient_set = np.array(list(patient_set))
        # subset_patients = np.random.choice(patient_set, 1000000, replace=False)
        # subset_patients = set(subset_patients)
        # meta_data_sub = self.meta_data[self.meta_data['enrol_id'].isin(subset_patients)]
        # enroll_sub = {k:v for k,v in self.enrolls.items() if k in subset_patients}
        # with open(os.path.join(root_dir, 'enrolls_sub.pkl'), 'wb') as f:
        #     pkl.dump(enroll_sub, f)
        # meta_data_sub.to_csv(os.path.join(root_dir, 'meta_data_sub.csv'), index=False)

        with open(os.path.join(root_dir, 'enrolls_sub.pkl'), 'rb') as f:
            self.enrolls = pkl.load(f)
        self.meta_data = self.meta_data[self.meta_data['enrol_id'].isin(self.enrolls.keys())]

        patient_set = set(self.meta_data['enrol_id'])
        patient_set = np.array(list(patient_set))
        train_idx, val_test_idx = train_test_split(patient_set, test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=random_state)
        if split == 'train':
            self.meta_data = self.meta_data[self.meta_data['enrol_id'].isin(train_idx)]
            self.patient_set = train_idx
        elif split == 'val':
            self.meta_data = self.meta_data[self.meta_data['enrol_id'].isin(val_idx)]
            self.patient_set = val_idx
        elif split == 'test':
            self.meta_data = self.meta_data[self.meta_data['enrol_id'].isin(test_idx)]
            self.patient_set = test_idx

        # self.enrolls = {k:v for k,v in self.enrolls.items() if k in self.patient_set}
    
    def __len__(self):
        return len(self.patient_set)
    
    def __getitem__(self, idx):
        patient = self.patient_set[idx]
        patient_meta = self.meta_data[self.meta_data['enrol_id'] == patient]
        patient_msa = patient_meta['MSA'].iloc[0]
        patient_enrolls = self.enrolls[patient]
        patient_meta = patient_meta[patient_meta['date'].isin(patient_enrolls.keys())]
        patient_dates = pd.to_datetime(patient_meta['date'].drop_duplicates()).dt.strftime('%Y-%m-%d')#.strftime('%m/%d/%Y')
        patient_enroll_codes = [patient_enrolls[date] for date in patient_dates]
        patient_enroll_codes = np.stack([self.encode_codelist(codes) for codes in patient_enroll_codes])
        patient_enroll_codes = torch.FloatTensor(patient_enroll_codes)

        input_sequence = patient_enroll_codes[:-1]
        target_sequence = self.get_target(patient_enroll_codes, patient_dates)
        if not self.use_all_seq:
            target_sequence = target_sequence[-1]

        ret_tuple = (input_sequence, target_sequence)
        if self.climate_data is not None:
            if 'year' in self.climate_data.columns:
                patient_year = pd.to_datetime(patient_dates).dt.year.iloc[-1]
                patient_climate = self.climate_data[(self.climate_data['GEOID'] == patient_msa)]
                patient_climate = patient_climate[patient_climate['year'] <= patient_year].sort_values(by='year').iloc[-1:]
                patient_climate = patient_climate.drop(columns=['GEOID', 'year'])
            else:
                patient_climate = self.climate_data[self.climate_data['GEOID'] == patient_msa]
                patient_climate = patient_climate.drop(columns=['GEOID'])
            patient_climate = patient_climate.to_numpy().reshape(-1)
            patient_climate = torch.FloatTensor(patient_climate)
            ret_tuple += (patient_climate,)
        if self.ret_date:
            if self.use_all_seq:
                if self.window_size is None:
                    target_dates = patient_dates.values[1:]
                else:
                    target_dates = patient_dates.values[:-1]
            else:
                if self.window_size is None:
                    target_dates = patient_dates.values[-1]
                else:
                    target_dates = patient_dates.values[-2]
            target_dates = np.array(target_dates)
            ret_tuple += (target_dates,)
        return ret_tuple
    
    def get_tokenizer(self):
        return {'cond_hist': self.tokenizer}
    
    def get_target(self, enroll_codes, dates):
        if self.window_size is None:
            return enroll_codes[1:]
        # dates = dates.reset_index(drop=True)
        date_codes = pd.DataFrame({'date': dates, 'codes': [x for x in enroll_codes]})
        date_codes['date'] = pd.to_datetime(date_codes['date'])
        date_codes['end_date'] = date_codes['date'] + pd.DateOffset(years=self.window_size)
        date_codes['before_date'] = date_codes['date'].apply(lambda x: date_codes[date_codes['date'] <= x]['codes'].sum())
        date_codes['target'] = date_codes['end_date'].apply(lambda x: date_codes[date_codes['date'] <= x]['codes'].sum())
        date_codes['target'] = date_codes['target'] - date_codes['before_date']
        target_sequence = np.array(date_codes['target'].to_list())[:-1]
        target_sequence = np.clip(target_sequence, 0, 1)
        target_sequence = torch.FloatTensor(target_sequence)
        return target_sequence

    def load_enrolls(self, meta_data):
        self.enrolls = {}
        file_list = set(meta_data['src'])
        eid_set = set(meta_data['enrol_id'])
        for file in file_list:
            with open(os.path.join(self.src_dir, file), 'r') as f:
                data = json.load(f)
            update_enrolls = {}
            for k,v in data.items():
                if k not in eid_set:
                    continue
                update_enrolls[k] = {}
                for date, codes in v.items():
                    if self.label_type == 'raw':
                        new_codes = [code for code in codes if code in self.code_vocabulary]
                    elif self.label_type == 'l2' or self.label_type == 'l1':
                        new_codes = [code for code in codes if code in self.icd_mapping]
                    if len(new_codes) == 0:
                        continue
                    update_enrolls[k][date] = new_codes
            self.enrolls.update(update_enrolls)

    def encode_codelist(self, codelist):
        out = np.zeros(self.code_vocabulary_size)
        if self.label_type == 'raw':
            codelist = [code for code in codelist if code in self.code_vocabulary]
        elif self.label_type == 'l2' or self.label_type == 'l1':
            codelist = [self.icd_mapping[code] for code in codelist if code in self.icd_mapping]
        codelist = list(set(codelist))
        # code_idx = np.array([self.code_mapping[code] for code in codelist if code in self.code_mapping])
        code_idx = np.array(self.tokenizer.convert_tokens_to_indices(codelist))
        if len(code_idx) > 0:
            out[code_idx] = 1
        return out


    def get_collate_fn(self):
        def collate_fn(batch):
            input_seq, target_seq, *others = zip(*batch)
            input_seq = pack_sequence(input_seq, enforce_sorted=False)
            target_seq = pack_sequence(target_seq, enforce_sorted=False) if self.use_all_seq else torch.stack(target_seq)
            
            ret_tuple = (input_seq, target_seq)
            if self.climate_data is not None:
                climate = others.pop(0)
                climate = torch.stack(climate)
                ret_tuple += (climate,)
            if self.ret_date:
                dates = others.pop(0)
                dates = list(dates) if self.use_all_seq else np.array(dates)
                ret_tuple += (dates,)
            return ret_tuple
        return collate_fn

        
if __name__ == '__main__':
    from tqdm import tqdm
    dataset = MarketScanDataset('data/processed/marketscan', split='train')
    print(len(dataset))
    # dataset[0]
    # for idx, item in enumerate(tqdm(dataset)):
    #     input_seq = item[0]
    #     num_codes = input_seq.sum(dim=1)
    #     if num_codes.min() == 0:
    #         print(idx)
    #         print(dataset.patient_set[idx])
    #         break
    #     pass