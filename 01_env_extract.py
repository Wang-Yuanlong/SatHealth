import ee
import pandas as pd
# import geopandas as gpd
import json
import numpy as np
import geemap
import os

ee.Authenticate()
PROJECT_NAME = '<put your google cloud project here>'
ee.Initialize(project=PROJECT_NAME)

geo_region = 'County' # 'County', 'ZCTA', 'CBSA', 'CT'

SELECTORS = {
    'ERA5-ECMWF' :  ['dewpoint_temperature_2m', 'temperature_2m', 
                     'soil_temperature_level_1', 'soil_temperature_level_3', 
                     'lake_bottom_temperature', 'lake_mix_layer_depth', 
                     'lake_mix_layer_temperature', 'lake_total_layer_temperature', 
                     'snow_albedo', 'snow_cover', 'snow_density', 'snow_depth', 
                     'skin_reservoir_content', 'volumetric_soil_water_layer_1', 
                     'volumetric_soil_water_layer_3', 'surface_latent_heat_flux_sum', 
                     'surface_net_solar_radiation_sum', 'surface_solar_radiation_downwards_sum',
                     'surface_thermal_radiation_downwards_sum', 'evaporation_from_bare_soil_sum', 
                     'evaporation_from_the_top_of_canopy_sum', 'evaporation_from_open_water_surfaces_excluding_oceans_sum',
                     'total_evaporation_sum', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
                     'surface_pressure', 'total_precipitation_sum', 'leaf_area_index_high_vegetation',
                     'leaf_area_index_low_vegetation', 'surface_runoff_sum'],
    'CAMS' :  ['total_aerosol_optical_depth_at_550nm_surface', 'particulate_matter_d_less_than_25_um_surface'],
    'S5' : ['NO2_column_number_density'],
    'OMI' : ['ozone'],
    'S2': ['NDVI', 'NDVI_binary']
}
SCALE = {
    'ERA5-ECMWF' : 1000,
    'CAMS' :  1000,
    'S5' : 1000,
    'OMI' : 1000,
    'S2': 50
}
IMAGE_SET = {
    'ERA5-ECMWF' : 'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    'S5': 'COPERNICUS/S5P/NRTI/L3_NO2',
    'OMI': 'TOMS/MERGED',
    'CAMS': 'ECMWF/CAMS/NRT',
    'S2': 'COPERNICUS/S2_SR_HARMONIZED'
}

def create_id_column(df, georegion, id_columns, timestamp_columns, variable_columns):
    df = df.copy()
    if georegion == 'County':
        df.drop(columns=[x for x in id_columns if x != 'COUNTYFP'], inplace=True)
        id_columns = ['COUNTYFP']
    elif georegion == 'ZCTA':
        df.drop(columns=[x for x in id_columns if x != 'ZCTA5'], inplace=True)
        id_columns = ['ZCTA5']
    elif georegion == 'CBSA':
        df.rename(columns={'GEOID': 'CBSAFP'}, inplace=True)
        id_columns = ['CBSAFP']
    elif georegion == 'CT':
        df['CTFP'] = df['GEO_ID'].apply(lambda x: x.split('US')[1])
        df.drop(columns=id_columns, inplace=True)
        id_columns = ['CTFP']

    df = df[id_columns + timestamp_columns + variable_columns]
    return df, id_columns

def efficent_fc_to_gdf(fc, selectors, batch_size=1000):
    # Get the total number of features in the feature collection
    num_features = fc.size().getInfo()
    # Define a list to store the GeoDataFrames
    gdfs = []
    # Loop over the feature collection in batches of size batch_size
    for i in range(0, num_features, batch_size):
        # Get the next batch of features using the limit() method
        # features = fc.limit(batch_size, str(i))
        features_list = fc.toList(batch_size, i)
        features = ee.FeatureCollection(features_list)

        # Convert the batch of features to a GeoDataFrame
        # gdf = geemap.ee_to_geopandas(features, selectors=selectors)
        gdf = geemap.ee_to_gdf(features, columns=selectors)

        # Add the GeoDataFrame to the list
        gdfs.append(gdf)

    # Concatenate the GeoDataFrames into a single GeoDataFrame
    final_gdf = pd.concat(gdfs)
    return final_gdf

def calculate_ndvi(image):
    # Calculate NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Define the threshold
    threshold = 0.2

    # Create a binary mask using the threshold
    ndvi_binary = ndvi.gt(threshold).toFloat().rename('NDVI_binary')

    filled_image = ndvi_binary.unmask(0)

    filled_image = filled_image.addBands(ndvi)
    return filled_image

asset_address = "data/georegions/county_ohio.geojson"
with open(asset_address) as f:
    asset = json.load(f)
ee_county_US = ee.FeatureCollection(asset)

asset_address = "data/georegions/zcta_ohio.geojson"
with open(asset_address) as f:
    asset = json.load(f)
ee_zip_US = ee.FeatureCollection(asset)

asset_address = "data/georegions/cbsa_ohio.geojson"
with open(asset_address) as f:
    asset = json.load(f)
ee_cbsa_US = ee.FeatureCollection(asset)

asset_address = "data/georegions/state_ohio.geojson"
with open(asset_address) as f:
    asset = json.load(f)
ee_state_US = ee.FeatureCollection(asset)

asset_address = "data/georegions/census_ohio.geojson"
with open(asset_address) as f:
    asset = json.load(f)
ee_census_US = ee.FeatureCollection(asset)


ee_state_US = ee_state_US.filter(ee.Filter.eq('STATEFP', '39'))
ee_county_US = ee_county_US.filter(ee.Filter.eq('STATEFP', '39'))
ee_zip_US = ee_zip_US.filterBounds(ee_state_US.first().geometry().buffer(-100)) # buffer to remove regions outside Ohio
ee_cbsa_US = ee_cbsa_US.filterBounds(ee_state_US.first().geometry().buffer(-100)) # buffer to remove regions outside Ohio

if geo_region == 'County':
    ee_roi = ee_county_US
    id_columns = ['GEOID', 'COUNTYFP']
elif geo_region == 'ZCTA':
    ee_roi = ee_zip_US
    id_columns = ['GEO_ID', 'ZCTA5']
elif geo_region == 'CBSA':
    ee_roi = ee_cbsa_US
    id_columns = ['GEOID']
elif geo_region == 'CT':
    ee_roi = ee_census_US
    id_columns = ['GEO_ID', 'COUNTY', 'TRACT']
num_areas = ee_roi.size().getInfo()

os.makedirs(f'data/raw/environment/{geo_region}/sub_files', exist_ok=True)
year_list = list(range(2016, 2023))
month_list = list(range(1, 13))
start_end_pairs = [(f"{year}-{month:02d}-01", f"{year}-{month + 1:02d}-01") if month != 12 else (f"{year}-{month:02d}-01", f"{year + 1}-01-01") for year in year_list for month in month_list]
all_data = {k:[] for k in IMAGE_SET}
for start_date, end_date in start_end_pairs:
    for image_set in IMAGE_SET:
        if os.path.exists(f'data/raw/environment/{geo_region}/sub_files/{image_set}_{start_date}_{end_date}.csv'):
            try:
                gdfs = pd.read_csv(f'data/raw/environment/{geo_region}/sub_files/{image_set}_{start_date}_{end_date}.csv')
            except pd.errors.EmptyDataError:
                gdfs = pd.DataFrame()
            all_data[image_set].append(gdfs)
            print(f"Finished {start_date} to {end_date} for {image_set}")
            continue
        im_dat = ee.ImageCollection(IMAGE_SET[image_set])
        gdfs, is_empty = [], True
        for i in range(0, num_areas, 100):
            ee_roi_sub = ee.FeatureCollection(ee_roi.toList(100, i))
            if image_set == 'CAMS' and start_date == '2023-05-01':
                # CAMS data lost on 2023-05-12
                im_dat1 = im_dat.filterDate('2023-05-01', '2023-05-12')
                im_dat2 = im_dat.filterDate('2023-05-13', '2023-06-01')
                filtered = im_dat1.merge(im_dat2).filterBounds(ee_roi_sub.geometry())
            elif image_set == 'S2':
                filtered = im_dat.filterDate(start_date, end_date).filterBounds(ee_roi_sub.geometry()).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 5))
                if filtered.size().getInfo() == 0:
                    print(f"No data for {start_date} to {end_date} for {image_set}")
                    continue
                filtered = calculate_ndvi(filtered.median())
                filtered = ee.ImageCollection([filtered])
            else:
                filtered = im_dat.filterDate(start_date, end_date).filterBounds(ee_roi_sub.geometry())
            
            if filtered.size().getInfo() == 0:
                print(f"No data for {start_date} to {end_date} for {image_set}")
                continue

            filtered = filtered.select(SELECTORS[image_set]).mean()
            means = filtered.reduceRegions(**{
                'collection': ee_roi_sub,  
                'reducer': ee.Reducer.mean(),
                'scale': SCALE[image_set],  
                'tileScale': 2
            })

            try:
                gdf = efficent_fc_to_gdf(means, id_columns + SELECTORS[image_set] if len(SELECTORS[image_set]) > 1 else id_columns + ['mean'])
            except ee.ee_exception.EEException as e:
                if 'internal error' in str(e):
                    print(f"internal error for {start_date} to {end_date} for {image_set}:\n{e}")
                    continue
                else:
                    raise e
            
            is_empty = False
            if len(SELECTORS[image_set]) == 1:
                gdf = gdf.rename(columns={'mean': SELECTORS[image_set][0]})
            gdfs.append(gdf)
        if not is_empty:
            gdfs = pd.concat(gdfs)
            gdfs['start_date'] = start_date
            gdfs['end_date'] = end_date
            gdfs, new_id_columns = create_id_column(gdfs, geo_region, id_columns, ['start_date', 'end_date'], SELECTORS[image_set])
        else:
            gdfs = pd.DataFrame()
        gdfs.to_csv(f'data/raw/environment/{geo_region}/sub_files/{image_set}_{start_date}_{end_date}.csv', index=False)
        all_data[image_set].append(gdfs)
        print(f"Finished {start_date} to {end_date} for {image_set}", flush=True)

for image_set in IMAGE_SET:
    all_data[image_set] = pd.concat(all_data[image_set])
    all_data[image_set].to_csv(f'data/raw/environment/{geo_region}/{image_set}.csv', index=False)

all_df = None
for v in all_data.values():
    v[new_id_columns] = v[new_id_columns].astype(str)
    if all_df is None:
        all_df = v
    else:
        if v.empty:
            continue
        all_df = all_df.merge(v, on=[*new_id_columns, 'start_date', 'end_date'], how='outer')
all_df.sort_values(new_id_columns + ['start_date', 'end_date'], inplace=True)
all_df.to_csv(f'data/raw/environment/{geo_region}/all_environment.csv', index=False)


landcover = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
selector = ['bare-coverfraction', 'urban-coverfraction', 'crops-coverfraction', 'grass-coverfraction', 'moss-coverfraction', 
            'water-permanent-coverfraction', 'water-seasonal-coverfraction', 'shrub-coverfraction', 'snow-coverfraction', 'tree-coverfraction']

lc_mean = landcover.select(selector).reduce(ee.Reducer.mean()).reduceRegions(ee_roi, reducer=ee.Reducer.mean(), scale=100)
lc_mean = lc_mean.getInfo()['features']
lc_mean = [feature['properties'] for feature in lc_mean]
lc_mean = pd.DataFrame(lc_mean).rename(columns={(k + '_mean'):k for k in selector})
lc_mean, _ = create_id_column(lc_mean, geo_region, id_columns, [], selector)
lc_mean.to_csv(f'data/raw/{geo_region}_landcover.csv', index=False)