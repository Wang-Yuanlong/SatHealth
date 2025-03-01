import os
import pandas as pd
import numpy as np
import pickle as pkl
import geopandas as gpd
from libpysal.weights import Queen
import json

georegion = 'County' # 'County', 'ZCTA', 'CBSA', 'CT'
data_dir = 'data'
georegion_data_file = os.path.join(data_dir, 'georegions')
raw_env_data_dir = f'data/raw/environment/{georegion}'
landcover_file = f'data/raw/{georegion}_landcover.csv'
img_feature_dir = os.path.join(data_dir, 'processed', 'gmap', f'img_features_{georegion}')
out_env_data_dir = os.path.join(data_dir, 'processed', georegion)
out_embedding_dir = f'data/embeddings/{georegion}'
years = list(range(2016, 2023))

with open('data/processed/gmap/img_features.json', 'r') as f:
    img_feature_map = json.load(f)

if georegion == 'County':
    id_column = 'COUNTYFP'
    georegion_data_file = os.path.join(georegion_data_file, 'county_ohio.geojson')
    region_id_column = 'COUNTYFP'
elif georegion == 'ZCTA':
    id_column = 'ZCTA5'
    georegion_data_file = os.path.join(georegion_data_file, 'zcta_ohio.geojson')
    region_id_column = 'ZCTA5'
elif georegion == 'CBSA':
    id_column = 'CBSAFP'
    georegion_data_file = os.path.join(georegion_data_file, 'cbsa_ohio.geojson')
    region_id_column = 'CBSAFP'
elif georegion == 'CT':
    id_column = 'CTFP'
    georegion_data_file = os.path.join(georegion_data_file, 'census_ohio.geojson')
    region_id_column = 'CTFP'
else:
    raise ValueError('Invalid georegion')

# process the raw environment data
region_shape = gpd.read_file(georegion_data_file)
if georegion == 'CT':
    region_shape['CTFP'] = region_shape['GEO_ID'].apply(lambda x: x.split('US')[1])
region_set = set(region_shape[region_id_column])

env = pd.read_csv(os.path.join(raw_env_data_dir, 'all_environment.csv'), dtype={id_column:str})
env = env[env[id_column].isin(region_set)]
env = env.sort_values(by=[id_column, 'start_date'])
env['start_date'] = pd.to_datetime(env['start_date'])
env['end_date'] = pd.to_datetime(env['end_date'])
env['year'] = env['start_date'].dt.year
env['month'] = env['start_date'].dt.month
env = env.drop(columns=['start_date', 'end_date'])
env = env.set_index([id_column, 'year', 'month']).reset_index()
env = env[env['year'].isin(years)]

landcover = pd.read_csv(landcover_file, dtype={id_column:str})
landcover = landcover[landcover[id_column].isin(region_set)].sort_values(by=id_column)

region_set = set(env[id_column]).intersection(set(landcover[id_column]))
env = env[env[id_column].isin(region_set)]
landcover = landcover[landcover[id_column].isin(region_set)]

climate_X_df = env.set_index([id_column, 'year', 'month'])

airquality_columns = ['total_aerosol_optical_depth_at_550nm_surface', 'particulate_matter_d_less_than_25_um_surface',
                      'NO2_column_number_density', 'ozone']
airquality_X_df = climate_X_df[airquality_columns].copy()
climate_X_df.drop(columns=airquality_columns, inplace=True)

greenary_columns = ['NDVI', 'NDVI_binary',
                    'leaf_area_index_high_vegetation',
                    'leaf_area_index_low_vegetation',]
greenary_X_df = climate_X_df[greenary_columns].copy()
climate_X_df.drop(columns=greenary_columns, inplace=True)

landcover_X_df = landcover.set_index([id_column])

os.makedirs(os.path.join(out_env_data_dir), exist_ok=True)
climate_X_df.to_csv(os.path.join(out_env_data_dir, 'climate.csv'))
greenary_X_df.to_csv(os.path.join(out_env_data_dir, 'greenery.csv'))
airquality_X_df.to_csv(os.path.join(out_env_data_dir, 'airquality.csv'))
landcover_X_df.to_csv(os.path.join(out_env_data_dir, 'landcover.csv'))


# create regional embeddings

region_shape = region_shape.set_index(region_id_column)
region_W = Queen.from_dataframe(region_shape, use_index=True)
region_W.transform = 'R'


region_image_features = {}
for region in region_set:
    if not os.path.exists(os.path.join(img_feature_dir, f'{region}.pkl')):
        continue
    with open(os.path.join(img_feature_dir, f'{region}.pkl'), 'rb') as f:
        region_image_features[region] = pkl.load(f)

region_set = set(region_image_features.keys()).intersection(region_set)
env = env[env[id_column].isin(region_set)]
landcover = landcover[landcover[id_column].isin(region_set)]
region_image_features = {k:v for k,v in region_image_features.items() if k in region_set}

env['season'] = ((env['month'] + 9) % 12) // 3
env = env.drop(columns=['month'])
env = env.groupby([id_column, 'year', 'season']).agg(lambda g: g.mean(skipna=True)).reset_index()

climate_X_df = env.pivot(index=[id_column, 'year'], columns='season')
climate_X_df.columns = [f'{col[0]}_{col[1]}' for col in climate_X_df.columns]
for col in climate_X_df.columns:
    if climate_X_df[col].std() == 0:
        climate_X_df.drop(columns=[col], inplace=True)
        continue
    climate_X_df[col] = (climate_X_df[col] - climate_X_df[col].mean()) / climate_X_df[col].std()
    if climate_X_df[col].isnull().sum() > 0:
        climate_X_df[col] = climate_X_df[col].groupby(id_column).transform(lambda x: x.fillna(x.mean()))

airquality_columns = ['total_aerosol_optical_depth_at_550nm_surface', 'particulate_matter_d_less_than_25_um_surface',
                      'NO2_column_number_density', 'ozone']
airquality_columns = [f'{col}_{season}' for col in airquality_columns for season in range(4)]
airquality_X_df = climate_X_df[airquality_columns].copy()
climate_X_df.drop(columns=airquality_columns, inplace=True)

greenary_columns = ['NDVI_0', 'NDVI_1', 'NDVI_2', 'NDVI_3', 'NDVI_binary_0', 'NDVI_binary_1', 'NDVI_binary_2', 'NDVI_binary_3',
                    'leaf_area_index_high_vegetation_0', 'leaf_area_index_high_vegetation_1', 
                    'leaf_area_index_high_vegetation_2', 'leaf_area_index_high_vegetation_3',
                    'leaf_area_index_low_vegetation_0', 'leaf_area_index_low_vegetation_1',
                    'leaf_area_index_low_vegetation_2', 'leaf_area_index_low_vegetation_3',]
greenary_X_df = climate_X_df[greenary_columns].copy()
climate_X_df.drop(columns=greenary_columns, inplace=True)

landcover_X_df = landcover.set_index([id_column])
for col in landcover_X_df.columns:
    if landcover_X_df[col].std() == 0:
        landcover_X_df.drop(columns=[col], inplace=True)
        continue
    landcover_X_df[col] = (landcover_X_df[col] - landcover_X_df[col].mean()) / landcover_X_df[col].std()
landcover_X_df = landcover_X_df.reset_index(names=[id_column])
landcover_X_df = env[[id_column, 'year']].drop_duplicates().merge(landcover_X_df).set_index([id_column, 'year'])

img_X_df = pd.DataFrame({region: region_image_features[region] for region in region_image_features}).T
for col in img_X_df.columns:
    if img_X_df[col].std() == 0:
        img_X_df.drop(columns=[col], inplace=True)
        continue
    img_X_df[col] = (img_X_df[col] - img_X_df[col].mean()) / img_X_df[col].std()
img_X_df = img_X_df.reset_index(names=[id_column])
img_X_df = env[[id_column, 'year']].drop_duplicates().merge(img_X_df).set_index([id_column, 'year'])
img_X_df.rename(columns=lambda x: img_feature_map[f'img_{x}'], inplace=True)

combined_X_df = pd.concat([climate_X_df, greenary_X_df, airquality_X_df, landcover_X_df, img_X_df], axis=1, join='inner')
combined_X_df = combined_X_df.dropna(how='any', axis=1)

os.makedirs(out_embedding_dir, exist_ok=True)
combined_X_df.to_csv(os.path.join(out_embedding_dir, f'embeddings.csv'))

def gen_history_features(df:pd.DataFrame, window=None, decay_factor=0.05):
    def decay_fn(x):
        return np.exp(- decay_factor * x)
    df = df.reset_index()
    df = df.sort_values(by=[id_column, 'year'])
    
    index_combination = df[[id_column, 'year']].merge(df[[id_column, 'year']], on=id_column, suffixes=('', '_history'))
    index_combination = index_combination[index_combination['year'] > index_combination['year_history']]
    index_combination['year_diff'] = index_combination['year'] - index_combination['year_history']
    if window is not None:
        index_combination = index_combination[index_combination['year_diff'] <= window]
    
    df_history = index_combination.merge(df, left_on=[id_column, 'year_history'], right_on=[id_column, 'year'], suffixes=('', '_r'), how='inner')
    df_history['decay'] = df_history['year_diff'].apply(decay_fn)
    df_history['decay'] = df_history['decay'] / df_history.groupby([id_column, 'year'])['decay'].transform('sum')
    df_history = df_history.drop(columns=['year_history', 'year_diff', 'year_r'])
    df_history = df_history.groupby([id_column, 'year'])[df_history.columns].apply(lambda g: (g.iloc[:, 2:-1].T * g['decay']).T.sum())
    return df_history

def gen_neighborhood_features(df:pd.DataFrame, W:Queen):
    df = df.reset_index()
    df[id_column] = df[id_column].astype(str)

    adjacent_pairs = [(i, j) for i in W.neighbors for j in W.neighbors[i]]
    index_combination = pd.DataFrame(adjacent_pairs, columns=[id_column, 'MSA_neighbor'])
    df_neighbor = index_combination.merge(df, left_on='MSA_neighbor', right_on=id_column, suffixes=('', '_r'))
    df_neighbor = df_neighbor.drop(columns=['MSA_neighbor', 'MSA_r'])
    df_neighbor = df_neighbor.groupby([id_column, 'year']).mean()
    return df_neighbor
    
combined_X_history_df = gen_history_features(combined_X_df, window=3)
combined_X_neighborhood_df = gen_neighborhood_features(combined_X_df, region_W)

combined_X_history_df.to_csv(os.path.join(out_embedding_dir, f'embeddings_T_agg.csv'))
combined_X_neighborhood_df.to_csv(os.path.join(out_embedding_dir, f'embeddings_S_agg.csv'))
