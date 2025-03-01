import os
import pandas as pd
import numpy as np
import pickle as pkl
import cv2
from multiprocessing import Pool
from tqdm import tqdm

georegion = 'County' # 'County', 'ZCTA', 'CBSA', 'CT'
satellite_meta_dir = f'data/processed'
satellite_img_dir = 'data/processed/images'
out_img_feature_dir = 'data/processed/gmap/img_features/'
out_region_feature_dir = f'data/processed/gmap/img_features_{georegion}/'
os.makedirs(out_img_feature_dir, exist_ok=True)
os.makedirs(out_region_feature_dir, exist_ok=True)

if georegion == 'County':
    id_column = 'COUNTYFP'
elif georegion == 'ZCTA':
    id_column = 'ZCTA5'
elif georegion == 'CBSA':
    id_column = 'CBSAFP'
elif georegion == 'CT':
    id_column = 'CTFP'
else:
    raise ValueError('Invalid georegion')

gmap_points = pd.read_csv(os.path.join(satellite_meta_dir, 'google_map_points.csv'), dtype={'COUNTYFP':str})
region_gmap_points = pd.read_csv(os.path.join(satellite_meta_dir, f'{georegion}', 'google_map_points_linked.csv'), dtype={id_column:str})

def get_stat_feature(array):
    array = array.flatten()
    return np.array([np.min(array), np.max(array), np.mean(array), np.std(array), np.median(array)])

def get_hist_feature(array, value_range):
    array = array.flatten()
    return np.histogram(array, bins=20, range=value_range)[0]

def get_img_feature(img):
    features = []
    B, G, R = img[...,0].astype(np.float32), img[...,1].astype(np.float32), img[...,2].astype(np.float32)
    B, G, R = B + 1e-6, G + 1e-6, R + 1e-6
    total = B + G + R
    b, g, r = B / total, G / total, R / total
    
    # RGB channel feature
    features.append(get_stat_feature(B))
    features.append(get_hist_feature(B, (0, 255))) # 25 features
    features.append(get_stat_feature(G))
    features.append(get_hist_feature(G, (0, 255))) # 25 features
    features.append(get_stat_feature(R))
    features.append(get_hist_feature(R, (0, 255))) # 25 features
    # Excess Green Index(ExG) = 2 * g - r - b (Normalized Excess Green Index (NExG))
    ExG = 2 * g - r - b
    features.append(get_stat_feature(ExG))
    features.append(get_hist_feature(ExG, (-1, 2)))
    # Excess Red Index(ExR) = 1.4 * r - g
    ExR = 1.4 * r - g
    features.append(get_stat_feature(ExR))
    features.append(get_hist_feature(ExR, (-1, 1.4)))
    # Excess Blue Index(ExB) = 1.4 * b - g
    ExB = 1.4 * b - g
    features.append(get_stat_feature(ExB))
    features.append(get_hist_feature(ExB, (-1, 1.4)))
    # Green Red Vegetation Index (GRVI) = (G-R)/(G+R)  (Normalized green-red difference index (NGRDI))
    GRVI = (G - R) / (G + R)
    features.append(get_stat_feature(GRVI))
    features.append(get_hist_feature(GRVI, (-1, 1)))
    # Modified Green Red Vegetation Index (MGRVI) = (G^2-R^2)/(G^2+R^2)
    MGRVI = (G**2 - R**2) / (G**2 + R**2)
    features.append(get_stat_feature(MGRVI))
    features.append(get_hist_feature(MGRVI, (-1, 1)))
    # Red Green Blue Vegetation Index (RGBVI) = (G^2-B*R)/(G^2+B*R)
    RGBVI = (G**2 - B*R) / (G**2 + B*R)
    features.append(get_stat_feature(RGBVI))
    features.append(get_hist_feature(RGBVI, (-1, 1)))
    # Kawashima Index = (R-B)/(R+B)
    Kawashima = (R - B) / (R + B)
    features.append(get_stat_feature(Kawashima))
    features.append(get_hist_feature(Kawashima, (-1, 1)))
    # Color index of vegetation extraction (CIVE) = 0.441r - 0.811g + 0.385b + 18.78745
    CIVE = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    features.append(get_stat_feature(CIVE))
    features.append(get_hist_feature(CIVE, (18.78745-0.811, 18.78745+0.441)))
    # Green leaf index (GLI) = (2g-r-b)/(2g+r+b)
    GLI = (2 * g - r - b) / (2 * g + r + b)
    features.append(get_stat_feature(GLI))
    features.append(get_hist_feature(GLI, (-1, 1)))

    features = np.concatenate(features) # 12 * 25 = 300 features
    return features

def gen_image_feature(img_filename, exist_skip=True):
    out_path = os.path.join(out_img_feature_dir, os.path.basename(img_filename).replace('.png', '.pkl'))
    if exist_skip and os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            return pkl.load(f)
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR) # img: (H, W, C=3), BGR
    img_feature = get_img_feature(img)
    with open(out_path, 'wb') as f:
        pkl.dump(img_feature, f)
    return img_feature

def get_region_img_feature(region_points):
    region_img_feature = []
    for row in region_points.itertuples():
        county, h, v = row.COUNTYFP, row.h, row.v
        if os.path.exists(os.path.join(out_img_feature_dir, f'gmap_{county}_{v}_{h}.pkl')):
            with open(os.path.join(out_img_feature_dir, f'gmap_{county}_{v}_{h}.pkl'), 'rb') as f:
                img_feature = pkl.load(f)
        else:
            img_fname = os.path.join(satellite_img_dir, f'gmap_{county}_{v}_{h}.png')
            if not os.path.exists(img_fname):
                continue
            img_feature = gen_image_feature(img_fname)
        region_img_feature.append(img_feature)
    if region_img_feature == []:
        return None
    region_img_feature = np.stack(region_img_feature, axis=0)
    region_img_feature = np.mean(region_img_feature, axis=0)
    return region_img_feature

def gen_region_image_feature(region_points, id_column):
    region_img_feature = get_region_img_feature(region_points)
    if region_img_feature is None:
        return None
    region_code = region_points[id_column].iloc[0]
    with open(os.path.join(out_region_feature_dir, f'{region_code}.pkl'), 'wb') as f:
        pkl.dump(region_img_feature, f)
    return region_img_feature

if __name__ == '__main__':
    num_workers = 16
    image_list = os.listdir(satellite_img_dir)

    # gen_image_feature(os.path.join(satellite_img_dir, image_list[0])) # test
    # gen_msa_image_feature(gmap_points[gmap_points['MSA'] == '18140']) # test
    # gen_zcta_image_feature(zcta_points[zcta_points['zcta5'] == '43017']) # test
    # gen_ct_image_feature(census_points[census_points['census_tract'] == '1400000US39001770100']) # test

    with Pool(num_workers) as p:
        myiter = p.imap_unordered(gen_image_feature, [os.path.join(satellite_img_dir, img) for img in image_list])
        for _ in tqdm(myiter, total=len(image_list)):
            pass

    region_list = gmap_points[id_column].unique()
    with Pool(num_workers) as p:
        myiter = p.imap_unordered(gen_region_image_feature, [region_gmap_points[region_gmap_points[id_column] == region] for region in region_list])
        for _ in tqdm(myiter, total=len(region_list)):
            pass
    
    import json
    image_feature_items = ['B', 'G', 'R', 'ExG', 'ExR', 'ExB', 'GRVI', 'MGRVI', 'RGBVI', 'Kawashima', 'CIVE', 'GLI']
    image_feature_stats = ['min', 'max', 'mean', 'std', 'median'] + [f'hist_{i}' for i in range(20)]
    image_feature_set = [f'{item}_{stat}' for item in image_feature_items for stat in image_feature_stats]
    image_feature_set = {f'img_{i}': f for i, f in enumerate(image_feature_set)}
    with open('data/processed/gmap/img_features.json', 'w') as f:
        json.dump(image_feature_set, f, indent=4)
        
    