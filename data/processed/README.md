# SatHealth Dataset
This is the dataset published with the paper

> Yuanlong Wang, Pengqi Wang, Changchang Yin, Ping Zhang
> 
> SatHealth: A Multimodal Public Health Dataset with Satellite-based Environmental Factors
> 
> Abstract: Living environments play a vital role in the prevalence and progression of diseases, and understanding their impact on patient’s health status becomes increasingly crucial for developing AI models. However, due to the lack of long-term and fine-grained spatial and temporal data in public and population health studies, most existing studies fail to incorporate environmental data, limiting the models’ performance and real-world application. To address this shortage, we developed SatHealth, a novel dataset combining multimodal temporal-spatial data, including environmental data, satellite images, all-disease prevalences, and social determinants of health (SDoH) indicators. We conducted experiments under two use cases with SatHealth: regional public health modeling and personal disease risk prediction. Experimental results show that living environmental information can significantly improve AI models’ performance and temporal-spatial generalizability on various tasks. Finally, we deploy a web-based application to provide an exploration tool for SatHealth and one-click access to both our data and regional environmental embedding to facilitate plug-and-play utilization. SatHealth is now published with data in all regions of Ohio, and we will keep updating SatHealth to cover the other parts of the US. With the web application and published code pipeline, our work provides valuable angles and resources to include environmental data in healthcare research and establishes a foundational framework for future research in environmental health informatics.

Detailed description of the dataset creation and baseline experiments can be found in the paper. Check out our web-based dataset explorer at [SatHealth Explorer](https://aimed-sathealth.net)

## File Structure

In the directory tree below, we use curly brackets ({}) to denote a copy for every item in the bracket.

```
.
├── {CBSA, County, CT, ZCTA}
│   ├── airquality.csv
│   ├── climate.csv
│   ├── greenery.csv
│   ├── landcover.csv
│   ├── google_map_points_linked.csv
│   └── sdi.csv                        # no CBSA version
├── column_dictionary.csv
├── icd{l1, l2, l3}_prev_ohio.csv
├── google_map_points.csv
├── google_map_request.py
└── README.md
```

SatHealth contains environmental and public health data in Ohio at multi-level geographic regions, including Core-based Statistical Areas(CBSAs), Counties, census tracts (CTs), and ZIP Code Tabulation Areas (ZCTAs). Data at different levels are packed into different folders.

`airquality.csv`, `climate.csv`, `greenery.csv`, `landcover.csv` are environmental data for corresponding regions. They are collected from multiple satellite products.

`google_map_points.csv` and `google_map_points_linked.csv` store the location metadata for requesting satellite images from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static), and `google_map_request.py` can be used to extract satellite images. See the [Satellite Image Extraction](#satellite-image-extraction) section below for details.

`sdi.csv` contains the [Social Deprivation Index (SDI)](https://www.graham-center.org/maps-data-tools/social-deprivation-index.html) for corresponding regions. We use the 2019 SDI in Ohio in SatHealth. Note that there is no SDI at the CBSA level.

`icd{l1, l2, l3}_prev_ohio.csv` contains the CBSA level disease prevalence of all diseases identified by the International Classification of Diseases, Tenth Revision, Clinical Modification (ICD-10-CM) codes. As ICD-10 codes have hierarchical structure corresponding to disease categories, we use `l1, l2, l3` to denote different levels of ICD codes used in the file, from the top level (`l1`) to the third level (`l3`). ICD-10-CM code definitions can be found [here](https://www.icd10data.com/).

## Column Description

`column_dictionary.csv` contains a full list of columns in all CSV files, along with their column type, data type, and description.

`column_type` in `column_dictionary.csv` denotes the role of columns in the table. It could be 

- `GEOID`, code to identify geographic regions. It is the FIPS code for CBSAs, counties, and census tracts; for ZCTAs, it is the region's ZIP code. 
- `timestamp`, which is used to mark the time for the record.
- `info`, other field in the table that provides information of the record.
- `variable`, data fields

`dtype` in `column_dictionary.csv` denotes the data type of the column. It could be 

- `string(n)`: string of length `n`, length is optional for variable length string type.
- `integer[m-n]`: integer ranges from `m` to `n`, inclusive. Value range may be absent.
- `double[m-n]`: decimal number ranges from `m` to `n`, inclusive. Value range may be absent.

`Unit` in `column_dictionary.csv` exists only for `variable` columns, denoting the unit of the value in this column.

## Satellite Image Extraction

> required package: tqdm, pandas

Due to copyright concerns, we do not include the satellite images we extracted from [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static). We provide our code to request satellite images from the Google API here. 

To extract satellite images in large batches to cover Ohio, we first create spatial grids with 500m stride and request images at grid points. The grid points we use are stored in `google_map_points.csv`. The script `google_map_request.py` will read this file and request satellite images.

To use `google_map_request.py`, you need a [Google Cloud](https://cloud.google.com) account. You will also need a project with a billing account for potential API usage fees.

To use Maps Static API, you need to 
1. [enable the API](https://developers.google.com/maps/documentation/maps-static/cloud-setup) in your project
2. [generate an API key](https://developers.google.com/maps/documentation/maps-static/get-api-key)
3. Fill your API key in `google_map_request.py` line 13.

By default, there are [usage limits](https://developers.google.com/maps/documentation/maps-static/usage-and-billing#other-usage-limits) for the API. Requests beyond the limit should be authenticated by digital signature. To enable requests with signatures in our script, you can set the `use_signature` at line 15 to `True` and fill in the signature key at line 14. You may find your signature key following Step 1 at [this page](https://developers.google.com/maps/documentation/maps-static/digital-signature).

The download process may take days depending on network speed. After downloading, the script will produce two files, `loc_meta.csv` and `err_meta.csv`. `loc_meta.csv` contains all images downloaded successfully and `err_meta.csv` contains failed requests with the error message. 

All images will be stored in the `images` folder within the same directory as the script. The naming rule is `gmap_{county}_{row_idx}_{col_idx}.png`.

`google_map_points_linked.csv` in the folders are not used for downloading. Instead, they provide a map from the regions to the set of points in `google_map_points.csv` to cover the region. We applied a buffer of 100m outside the region boundary when calculating the containment relation between the region and the point. As a result, a single image may be included in multiple regions.
