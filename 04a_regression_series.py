import os
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
import geopandas as gpd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from multiprocessing import Pool

icd_level = 'l1' # 'l1', 'l2', 'l3'
marketscan_dir = 'data/processed'
icd10_dir = 'data/icd10/'
embedding_dir = 'data/embeddings/CBSA'
result_dir = 'results'
regression_model = 'RF'
os.makedirs(f'results/regression_{regression_model}', exist_ok=True)
os.makedirs(f'results/regression_sdi_{regression_model}', exist_ok=True)
MSA2code = {'Wheeling':'48540', 'Huntington-Ashland':'26580', 'Cincinnati':'17140', 
            'Cleveland-Elyria':'17460', 'Akron':'10420', 'Canton-Massillon':'15940', 
            'Columbus':'18140', 'Dayton':'19380', 'Springfield':'44220', 'Lima':'30620', 
            'Mansfield':'31900', 'Weirton-Steubenville':'48260', 'Toledo':'45780', 
            'Youngstown-Warren-Boardman':'49660', 'Athens':'11900', 'Point Pleasant':'38580', 
            'Coshocton':'18740', 'Jackson':'27160', 'Wooster':'49300', 'Fremont':'23380', 
            'Defiance':'19580', 'Portsmouth':'39020', 'Wilmington':'48940', 'Sandusky':'41780', 
            'Ashtabula':'11780', 'New Philadelphia-Dover':'35420', 'Norwalk':'35940', 
            'Bellefontaine':'13340', 'Chillicothe':'17060', 'Zanesville':'49780', 'Cambridge':'15740', 
            'Marion':'32020', 'Mount Vernon':'34540', 'Washington Court House':'47920', 
            'Urbana':'46500', 'Greenville':'24820', 'Sidney':'43380', 'Findlay':'22300', 
            'Tiffin':'45660', 'Van Wert':'46780', 'Celina':'16380', 'Wapakoneta':'47540', 
            'Bucyrus':'15340', 'Ashland':'11740', 'Marietta':'31930', 'Port Clinton':'38840', 'Salem':'41400'}
code2MSA = {v: k for k, v in MSA2code.items()}

with open(os.path.join(icd10_dir, f'icd10{icd_level}.json'), 'r') as f:
    icd10 = json.load(f)
with open('data/processed/gmap/img_features.json', 'r') as f:
    img_feature_map = json.load(f)

rf_kwargs_list = [{'n_estimators':n_estimators, 'random_state':random_state} 
                  for n_estimators in [150, 200] for random_state in [0]]
histgbr_kwargs_list = [{'n_estimators':n_estimators, 'random_state':random_state} 
                        for n_estimators in [150, 200] for random_state in [0]]
lr_kwargs_list = [{'n_estimators':100, 'random_state':0}]
kwargs_list = {'RF':rf_kwargs_list, 'HGBR':histgbr_kwargs_list, 'LR':lr_kwargs_list}

def regression_exp(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators=100, random_state=0, hist=None, neib=None):
    if hist is not None and isinstance(hist, pd.DataFrame):
        hist_train, hist_test = hist.loc[X_train.index], hist.loc[X_test.index]
        hist_train = hist_train.values
        hist_test = hist_test.values
    if neib is not None and isinstance(neib, pd.DataFrame):
        neib_train, neib_test = neib.loc[X_train.index], neib.loc[X_test.index]
        neib_train = neib_train.values
        neib_test = neib_test.values
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    Y_train_ = Y_train[code].values
    Y_test_ = Y_test[code].values

    if regression_model == 'LR':
        model = LinearRegression()
        if hist is not None:
            model_hist = LinearRegression()
        if neib is not None:
            model_neib = LinearRegression()
    elif regression_model == 'RF':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if hist is not None:
            model_hist = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if neib is not None:
            model_neib = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    elif regression_model == 'HGBR':
        model = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=random_state)
        if hist is not None:
            model_hist = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=random_state)
        if neib is not None:
            model_neib = HistGradientBoostingRegressor(max_iter=n_estimators, random_state=random_state)
    
    model.fit(X_train, Y_train_)
    Y_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    if hist is not None:
        Y_train_pred = Y_train_pred.copy()
        model_hist.fit(hist_train, Y_train_ - Y_train_pred)
        Y_train_pred += model_hist.predict(hist_train)
        Y_pred += model_hist.predict(hist_test)
    if neib is not None:
        Y_train_pred = Y_train_pred.copy()
        model_neib.fit(neib_train, Y_train_ - Y_train_pred)
        Y_train_pred += model_neib.predict(neib_train)
        Y_pred += model_neib.predict(neib_test)

    mse = mean_squared_error(Y_test_, Y_pred)
    mae = mean_absolute_error(Y_test_, Y_pred)
    r2 = r2_score(Y_test_, Y_pred)
    ret = {'setting':X_name, 'fold': fold_idx, 'code':code, 'mse':mse, 'mae':mae, 'r2':r2, 'n_estimators':n_estimators, 'random_state':random_state}
    return ret

def split_combined_df(combined_X_df):
    climate_X_df = combined_X_df.copy()
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

    landcover_columns = ['bare-coverfraction', 'urban-coverfraction', 'crops-coverfraction', 'grass-coverfraction', 'moss-coverfraction', 
                'water-permanent-coverfraction', 'water-seasonal-coverfraction', 'shrub-coverfraction', 'tree-coverfraction']
    landcover_X_df = climate_X_df[landcover_columns].copy()
    climate_X_df.drop(columns=landcover_columns, inplace=True)

    image_columns = [v for k, v in img_feature_map.items()]
    img_X_df = climate_X_df[image_columns].copy()
    climate_X_df.drop(columns=image_columns, inplace=True)
    return climate_X_df, greenary_X_df, airquality_X_df, landcover_X_df, img_X_df

# ICD regression
combined_X_df = pd.read_csv(os.path.join(embedding_dir, 'embeddings.csv'), dtype={'CBSAFP':str})

market_scan = pd.read_csv(os.path.join(marketscan_dir, f'icd{icd_level}_prev_ohio.csv'), dtype={'code':str, 'CBSAFP':str})
market_scan['year'] = market_scan['year'].astype(int)
code_count = market_scan.groupby(['code'])['count'].sum()
patient_count_msa = market_scan[['CBSAFP', 'year', 'count_patient']].drop_duplicates()
patient_count_msa = patient_count_msa.groupby(['CBSAFP'])['count_patient'].mean()
market_scan = market_scan[(market_scan['code'].isin(code_count[code_count>1000].index))
                          &(market_scan['code'].isin(icd10.keys()))
                          &(market_scan['CBSAFP'].isin(patient_count_msa[patient_count_msa>100].index))].sort_values(by=['CBSAFP', 'code', 'year'])

market_scan_Y_df = market_scan.pivot(index=['CBSAFP', 'year'], columns='code', values='prevalence')
market_scan_Y_df.columns.name = None
for col in market_scan_Y_df.columns:
    if market_scan_Y_df[col].std() == 0:
        market_scan_Y_df.drop(columns=[col], inplace=True)
        continue
    market_scan_Y_df[col] = (market_scan_Y_df[col] - market_scan_Y_df[col].mean()) / market_scan_Y_df[col].std()
market_scan_Y_df = market_scan_Y_df.reset_index()

index_set = combined_X_df[['CBSAFP', 'year']].merge(market_scan_Y_df[['CBSAFP', 'year']], on=['CBSAFP', 'year'], how='inner').set_index(['CBSAFP', 'year'])
combined_X_df = combined_X_df.set_index(['CBSAFP', 'year']).loc[index_set.index]
market_scan_Y_df = market_scan_Y_df.set_index(['CBSAFP', 'year']).loc[index_set.index]
climate_X_df, greenary_X_df, airquality_X_df, landcover_X_df, img_X_df = split_combined_df(combined_X_df)

env_X_df = pd.concat([climate_X_df, greenary_X_df, airquality_X_df], axis=1, join='inner')
X_settings = [climate_X_df, env_X_df, landcover_X_df, img_X_df, combined_X_df]
X_settings_name = ['climate', 'env', 'landcover', 'img', 'combined']

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
regression_result_df = []
regression_result_spatial_df = []
regression_result_temporal_df = []
for X_df, X_name in zip(X_settings, X_settings_name):
    print(f'X: {X_name}')
    MSAs = X_df.index.unique(level=0)
    years = X_df.index.unique(level=1)
    train_years = years[:4]
    test_years = years[4:]
    for kwargs in kwargs_list[regression_model]:
        n_estimators = kwargs['n_estimators']
        random_state = kwargs['random_state']
        # random split
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(X_df.index)):
            train_index = X_df.index[train_index]
            test_index = X_df.index[test_index]
            X_train, X_test = X_df.loc[train_index], X_df.loc[test_index]
            Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state) for code in Y_train.columns])
            regression_result, _, _ = map(list, zip(*regression_result))
            regression_result_df += regression_result
        # spatial split
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(MSAs)):
            train_index = MSAs[train_index]
            test_index = MSAs[test_index]
            X_train, X_test = X_df.loc[train_index], X_df.loc[test_index]
            Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatial_df += regression_result
        
        # temporal split
        X_train, X_test = X_df.loc[(slice(None), train_years), :], X_df.loc[(slice(None), test_years), :]
        Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
        with Pool(16) as p:
            regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, 0, X_name, n_estimators, random_state) for code in Y_train.columns])
        regression_result = list(regression_result)
        regression_result_temporal_df += regression_result
        
regression_result_df = pd.DataFrame(regression_result_df)
regression_result_df.to_csv(f'results/regression_{regression_model}/icd_regression_result_{icd_level}.csv', index=False)
regression_result_spatial_df = pd.DataFrame(regression_result_spatial_df)
regression_result_spatial_df.to_csv(f'results/regression_{regression_model}/icd_regression_result_spatial_{icd_level}.csv', index=False)
regression_result_temporal_df = pd.DataFrame(regression_result_temporal_df)
regression_result_temporal_df.to_csv(f'results/regression_{regression_model}/icd_regression_result_temporal_{icd_level}.csv', index=False)

combined_X_hist_df = pd.read_csv(os.path.join(embedding_dir, 'embeddings_T_agg.csv'), dtype={'CBSAFP':str})
combined_X_neib_df = pd.read_csv(os.path.join(embedding_dir, 'embeddings_S_agg.csv'), dtype={'CBSAFP':str})
index_set = index_set.reset_index().merge(combined_X_hist_df[['CBSAFP', 'year']], on=['CBSAFP', 'year'], how='inner')
index_set = index_set.merge(combined_X_neib_df[['CBSAFP', 'year']], on=['CBSAFP', 'year'], how='inner').set_index(['CBSAFP', 'year'])
combined_X_df = combined_X_df.loc[index_set.index]
market_scan_Y_df = market_scan_Y_df.loc[index_set.index]
combined_X_hist_df.set_index(['CBSAFP', 'year']).loc[index_set.index]
combined_X_neib_df.set_index(['CBSAFP', 'year']).loc[index_set.index]
climate_X_hist_df, greenary_X_hist_df, airquality_X_hist_df, landcover_X_hist_df, img_X_hist_df = split_combined_df(combined_X_hist_df)
climate_X_neib_df, greenary_X_neib_df, airquality_neib_X_df, landcover_neib_X_df, img_neib_X_df = split_combined_df(combined_X_neib_df)


env_X_df = pd.concat([climate_X_df, greenary_X_df, airquality_X_df], axis=1, join='inner')
env_X_hist_df = pd.concat([climate_X_hist_df, greenary_X_hist_df, airquality_X_hist_df], axis=1, join='inner')
env_X_neib_df = pd.concat([climate_X_neib_df, greenary_X_neib_df, airquality_neib_X_df], axis=1, join='inner')
X_settings = [(combined_X_df, combined_X_hist_df, combined_X_neib_df),
              (combined_X_df, combined_X_hist_df),
              (combined_X_df, combined_X_neib_df),
              (env_X_df, env_X_hist_df, env_X_neib_df),
              (env_X_df, env_X_hist_df),
              (env_X_df, env_X_neib_df),]
X_settings_name = ['All+T+S', 'All+T', 'All+S', 'Env+T+S', 'Env+T', 'Env+S']

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
regression_result_spatiali_df = []
regression_result_spatiale_df = []
regression_result_temporal_df = []
for X_dfs, X_name in zip(X_settings, X_settings_name):
    print(f'X: {X_name}')
    if X_name.endswith('+T+S'):
        X_df, X_hist_df, X_neib_df = X_dfs
        common_index = X_df.index.intersection(X_hist_df.index).intersection(X_neib_df.index)
        X_df = X_df.loc[common_index]
        X_hist_df = X_hist_df.loc[common_index]
        X_neib_df = X_neib_df.loc[common_index]
    elif X_name.endswith('+T'):
        X_df, X_hist_df = X_dfs
        common_index = X_df.index.intersection(X_hist_df.index)
        X_df = X_df.loc[common_index]
        X_hist_df = X_hist_df.loc[common_index]
        X_neib_df = None
    elif X_name.endswith('+S'):
        X_df, X_neib_df = X_dfs
        common_index = X_df.index.intersection(X_neib_df.index)
        X_df = X_df.loc[common_index]
        X_neib_df = X_neib_df.loc[common_index]
        X_hist_df = None
    else:
        X_df = X_dfs[0]
        X_hist_df = None
        X_neib_df = None

    msa_regions = gpd.read_file('data/raw/georegions/cbsa_ohio.geojson')
    msa_regions['GEOID'] = msa_regions['GEOID'].astype(int)
    msa_regions = msa_regions[msa_regions['GEOID'].isin(X_df.index.get_level_values(0).unique())]
    msa_regions = msa_regions.set_index('GEOID')
    msa_regions.to_crs('EPSG:32617', inplace=True) # Ohio
    msa_regions['centroid'] = msa_regions.geometry.centroid
    msa_regions['x'] = msa_regions.centroid.x
    msa_regions['y'] = msa_regions.centroid.y
    kmeans = KMeans(n_clusters=5, random_state=42)
    msa_regions['cluster'] = kmeans.fit_predict(msa_regions[['x', 'y']])

    split_list = []
    for cluster in msa_regions['cluster'].unique():
        train_idx = msa_regions[msa_regions['cluster'] != cluster].index
        test_idx = msa_regions[msa_regions['cluster'] == cluster].index
        split_list.append({'train_idx':train_idx, 'test_idx':test_idx})


    MSAs = X_df.index.unique(level=0)
    years = X_df.index.unique(level=1)
    train_years = years[:4]
    test_years = years[4:]
    for kwargs in kwargs_list[regression_model]:
        n_estimators = kwargs['n_estimators']
        random_state = kwargs['random_state']
        # spatial i split
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(MSAs)):
            train_index = MSAs[train_index]
            test_index = MSAs[test_index]
            X_train, X_test = X_df.loc[train_index], X_df.loc[test_index]
            Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state, X_hist_df, X_neib_df) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatiali_df += regression_result
        
        # spatial e split
        for fold_idx, split in enumerate(split_list):
            train_idx, test_idx = split['train_idx'], split['test_idx']
            X_train, X_test = X_df.loc[train_idx], X_df.loc[test_idx]
            Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state, X_hist_df, X_neib_df) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatiale_df += regression_result
        
        # temporal split
        X_train, X_test = X_df.loc[(slice(None), train_years), :], X_df.loc[(slice(None), test_years), :]
        Y_train, Y_test = market_scan_Y_df.loc[X_train.index], market_scan_Y_df.loc[X_test.index]
        with Pool(16) as p:
            regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, 0, X_name, n_estimators, random_state, X_hist_df, X_neib_df) for code in Y_train.columns])
        regression_result = list(regression_result)
        regression_result_temporal_df += regression_result

os.makedirs(f'results/regression_{regression_model}/hist_neib', exist_ok=True)
regression_result_spatiali_df = pd.DataFrame(regression_result_spatiali_df)
regression_result_spatiali_df.to_csv(f'results/regression_{regression_model}/hist_neib/icd_regression_result_spatiali_{icd_level}.csv', index=False)
regression_result_spatiale_df = pd.DataFrame(regression_result_spatiale_df)
regression_result_spatiale_df.to_csv(f'results/regression_{regression_model}/hist_neib/icd_regression_result_spatiale_{icd_level}.csv', index=False)
regression_result_temporal_df = pd.DataFrame(regression_result_temporal_df)
regression_result_temporal_df.to_csv(f'results/regression_{regression_model}/hist_neib/icd_regression_result_temporal_{icd_level}.csv', index=False)


# SDI regression
sdi_Y_df = pd.read_csv(f'data/processed/ZCTA/sdi.csv').dropna(how='any', axis=1).rename(columns={'ZCTA5_FIPS':'ZCTA5'})
sdi_Y_df = sdi_Y_df[['ZCTA5', 'sdi','pct_Poverty_LT100','pct_Single_Parent_Fam',
                     'pct_Education_LT12years','pct_NonEmployed','pctHH_No_Vehicle',
                     'pctHH_Renter_Occupied','pctHH_Crowding']]

combined_X_df = pd.read_csv(os.path.join(embedding_dir, '../ZCTA', 'embeddings.csv'), dtype={'ZCTA5':str})
combined_X_df = combined_X_df[(combined_X_df['year'] == 2019) 
                            & (combined_X_df['ZCTA5'].isin(sdi_Y_df['ZCTA5']))]  \
                .groupby('ZCTA5').mean()
sdi_Y_df = sdi_Y_df.set_index('ZCTA5').loc[combined_X_df.index]
climate_X_df, greenary_X_df, airquality_X_df, landcover_X_df, img_X_df = split_combined_df(combined_X_df)

env_X_df = pd.concat([climate_X_df, greenary_X_df, airquality_X_df, landcover_X_df], axis=1, join='inner')
X_settings = [climate_X_df, env_X_df, landcover_X_df, img_X_df, combined_X_df]
X_settings_name = ['climate', 'env', 'landcover', 'img', 'combined']
    
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
regression_result_spatial_df = []
for X_df, X_name in zip(X_settings, X_settings_name):
    print(f'X: {X_name}')
    MSAs = X_df.index.unique(level=0)
    for kwargs in kwargs_list[regression_model]:
        n_estimators = kwargs['n_estimators']
        random_state = kwargs['random_state']
        # spatial split
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(MSAs)):
            train_index = MSAs[train_index]
            test_index = MSAs[test_index]
            X_train, X_test = X_df.loc[train_index], X_df.loc[test_index]
            Y_train, Y_test = sdi_Y_df.loc[X_train.index], sdi_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatial_df += regression_result
        
regression_result_spatial_df = pd.DataFrame(regression_result_spatial_df)
regression_result_spatial_df.to_csv(f'results/regression_sdi_{regression_model}/sdi_regression_result_spatial.csv', index=False)

combined_X_hist_df = pd.read_csv(os.path.join(embedding_dir, '../ZCTA', 'embeddings_T_agg.csv'), dtype={'ZCTA5':str})
combined_X_neib_df = pd.read_csv(os.path.join(embedding_dir, '../ZCTA', 'embeddings_S_agg.csv'), dtype={'ZCTA5':str})
combined_X_hist_df = combined_X_hist_df[(combined_X_hist_df['year'] == 2019) 
                            & (combined_X_hist_df['ZCTA5'].isin(sdi_Y_df['ZCTA5']))]  \
                .groupby('ZCTA5').mean()
combined_X_neib_df = combined_X_neib_df[(combined_X_neib_df['year'] == 2019)
                                      & (combined_X_neib_df['ZCTA5'].isin(sdi_Y_df['ZCTA5']))]  \
                     .groupby('ZCTA5').mean()

index_set = combined_X_df.reset_index()[['ZCTA5']].merge(combined_X_hist_df[['ZCTA5']], on=['ZCTA5'], how='inner')
index_set = index_set.merge(combined_X_neib_df[['ZCTA5']], on=['ZCTA5'], how='inner').set_index('ZCTA5')
combined_X_df = combined_X_df.loc[index_set.index]
sdi_Y_df = sdi_Y_df.loc[index_set.index]
combined_X_hist_df.loc[index_set.index]
combined_X_neib_df.loc[index_set.index]
climate_X_hist_df, greenary_X_hist_df, airquality_X_hist_df, landcover_X_hist_df, img_X_hist_df = split_combined_df(combined_X_hist_df)
climate_X_neib_df, greenary_X_neib_df, airquality_neib_X_df, landcover_neib_X_df, img_neib_X_df = split_combined_df(combined_X_neib_df)

env_X_df = pd.concat([climate_X_df, greenary_X_df, airquality_X_df], axis=1, join='inner')
env_X_neib_df = pd.concat([climate_X_neib_df, greenary_X_neib_df, airquality_neib_X_df], axis=1, join='inner')
env_X_hist_df = pd.concat([climate_X_hist_df, greenary_X_hist_df, airquality_X_hist_df], axis=1, join='inner')

X_settings = [(combined_X_df, combined_X_hist_df, combined_X_neib_df),
              (combined_X_df, combined_X_neib_df),
              (combined_X_df, combined_X_hist_df),
              (combined_X_df,),
              (env_X_df, env_X_hist_df, env_X_neib_df),
              (env_X_df, env_X_hist_df),
              (env_X_df, env_X_neib_df),
              (env_X_df,)]

X_settings_name = ['All+T+S', 'All+S', 'All+T', 'All', 'Env+T+S', 'Env+T', 'Env+S', 'Env']

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
regression_result_spatiali_df = []
regression_result_spatiale_df = []

for X_dfs, X_name in zip(X_settings, X_settings_name):
    print(f'X: {X_name}')
    if X_name.endswith('+T+S'):
        X_df, X_hist_df, X_neib_df = X_dfs
        common_index = X_df.index.intersection(X_hist_df.index).intersection(X_neib_df.index)
        X_df = X_df.loc[common_index]
        X_hist_df = X_hist_df.loc[common_index]
        X_neib_df = X_neib_df.loc[common_index]
    elif X_name.endswith('+S'):
        X_df, X_neib_df = X_dfs
        common_index = X_df.index.intersection(X_neib_df.index)
        X_df = X_df.loc[common_index]
        X_neib_df = X_neib_df.loc[common_index]
        X_hist_df = None
    elif X_name.endswith('+T'):
        X_df, X_hist_df = X_dfs
        common_index = X_df.index.intersection(X_hist_df.index)
        X_df = X_df.loc[common_index]
        X_hist_df = X_hist_df.loc[common_index]
        X_neib_df = None
    else:
        X_df = X_dfs[0]
        X_hist_df = None
        X_neib_df = None

    zcta_regions = gpd.read_file('data/georegions/zcta_ohio.geojson')
    zcta_regions['ZCTA5'] = zcta_regions['ZCTA5'].astype(str)
    zcta_regions = zcta_regions[zcta_regions['ZCTA5'].isin(X_df.index)]
    zcta_regions = zcta_regions.set_index('ZCTA5')
    zcta_regions.to_crs('EPSG:32617', inplace=True) # Ohio
    zcta_regions['centroid'] = zcta_regions.geometry.centroid
    zcta_regions['x'] = zcta_regions.centroid.x
    zcta_regions['y'] = zcta_regions.centroid.y

    kmeans = KMeans(n_clusters=5, random_state=42)
    zcta_regions['cluster'] = kmeans.fit_predict(zcta_regions[['x', 'y']])

    split_list = []
    for cluster in zcta_regions['cluster'].unique():
        train_idx = zcta_regions[zcta_regions['cluster'] != cluster].index
        test_idx = zcta_regions[zcta_regions['cluster'] == cluster].index
        split_list.append({'train_idx':train_idx, 'test_idx':test_idx})

    ZCTAs = X_df.index.unique(level=0)
    for kwargs in kwargs_list[regression_model]:
        n_estimators = kwargs['n_estimators']
        random_state = kwargs['random_state']
        # spatial i split
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(ZCTAs)):
            train_index = ZCTAs[train_index]
            test_index = ZCTAs[test_index]
            X_train, X_test = X_df.loc[train_index], X_df.loc[test_index]
            Y_train, Y_test = sdi_Y_df.loc[X_train.index], sdi_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state, X_hist_df, X_neib_df) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatiali_df += regression_result

        # spatial e split
        for fold_idx, split in enumerate(split_list):
            train_idx, test_idx = split['train_idx'], split['test_idx']
            X_train, X_test = X_df.loc[train_idx], X_df.loc[test_idx]
            Y_train, Y_test = sdi_Y_df.loc[X_train.index], sdi_Y_df.loc[X_test.index]
            with Pool(16) as p:
                regression_result = p.starmap(regression_exp, [(X_train, X_test, Y_train, Y_test, code, fold_idx, X_name, n_estimators, random_state, X_hist_df, X_neib_df) for code in Y_train.columns])
            regression_result = list(regression_result)
            regression_result_spatiale_df += regression_result

os.makedirs(f'results/regression_sdi_{regression_model}/hist_neib', exist_ok=True)
regression_result_spatiali_df = pd.DataFrame(regression_result_spatiali_df)
regression_result_spatiali_df.to_csv(f'results/regression_sdi_{regression_model}/hist_neib/sdi_regression_result_spatiali_tmp.csv', index=False)
regression_result_spatiale_df = pd.DataFrame(regression_result_spatiale_df)
regression_result_spatiale_df.to_csv(f'results/regression_sdi_{regression_model}/hist_neib/sdi_regression_result_spatiale_tmp.csv', index=False)
              

