import os
import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr, kendalltau, ttest_ind, fisher_exact
from scipy.stats.contingency import odds_ratio

icd_level = 'l1' # 'l1', 'l2', 'l3'
icd10_dir = 'data/icd10'
marketscan_dir = 'data/processed'
embedding_dir = 'data/embeddings/CBSA'
result_dir = 'results/stat_results'
os.makedirs(result_dir, exist_ok=True)

with open(os.path.join(icd10_dir, f'icd10{icd_level}.json'), 'r') as f:
    icd10 = json.load(f)

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

# pairwise correlation
xy_corr_df = []
for f_idx, feature in enumerate(combined_X_df.columns):
    if feature in market_scan_Y_df.columns:
        continue
    X_vec = combined_X_df[feature].values
    for c_idx, code in enumerate(market_scan_Y_df.columns):
        Y_vec = market_scan_Y_df[code].values
        spearman_r, spearman_pval = spearmanr(X_vec, Y_vec)
        kendall_tau, kendall_pval = kendalltau(X_vec, Y_vec)
        xy_corr_df.append({'feature':feature, 'code':code, 'spearman_r':spearman_r, 'spearman_pval':spearman_pval,
                            'kendall_tau':kendall_tau, 'kendall_pval':kendall_pval})
xy_corr_df = pd.DataFrame(xy_corr_df)
xy_corr_df.to_csv(os.path.join(result_dir, f'xy_corr_{icd_level}.csv'), index=False)


# t-test

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
Urban_Y_df = market_scan_Y_df.loc[[MSA2code['Columbus'], MSA2code['Cincinnati'], MSA2code['Cleveland-Elyria']]]
Rural_Y_df = market_scan_Y_df.drop(index=[MSA2code['Columbus'], MSA2code['Cincinnati'], MSA2code['Cleveland-Elyria']])

ru_code_ttest_df = []
for code in market_scan_Y_df.columns:
    ttest_result = ttest_ind(Urban_Y_df[code].values, Rural_Y_df[code].values, equal_var=False)
    ru_code_ttest_df.append({'code':code, 't':ttest_result.statistic, 'pval':ttest_result.pvalue})
ru_code_ttest_df = pd.DataFrame(ru_code_ttest_df)
ru_code_ttest_df = ru_code_ttest_df[ru_code_ttest_df['pval'] < 0.01]
ru_code_ttest_df['description'] = ru_code_ttest_df['code'].apply(lambda x: icd10[x]['description'])
ru_code_ttest_df['pval_str'] = ru_code_ttest_df['pval'].apply(lambda x: f'{x:.5f}')
ru_code_ttest_df['t_str'] = ru_code_ttest_df['t'].apply(lambda x: f'{x:.5f}')
ru_code_ttest_df = ru_code_ttest_df[['code', 't_str', 'pval_str', 'description', 't', 'pval']].sort_values(by='pval')
ru_code_ttest_df.to_csv(os.path.join(result_dir, f'ru_code_ttest_{icd_level}.csv'), index=False)


# odd ratio
market_scan_urban = market_scan[market_scan['CBSAFP'].isin([MSA2code['Columbus'], MSA2code['Cincinnati'], MSA2code['Cleveland-Elyria']])]
market_scan_rural = market_scan[~market_scan['CBSAFP'].isin([MSA2code['Columbus'], MSA2code['Cincinnati'], MSA2code['Cleveland-Elyria']])]

market_scan_urban = market_scan_urban[['code', 'CBSAFP', 'year', 'count', 'count_patient']].groupby(['code', 'year']).agg({'count':'sum', 'count_patient':'sum'}).reset_index()
market_scan_rural = market_scan_rural[['code', 'CBSAFP', 'year', 'count', 'count_patient']].groupby(['code', 'year']).agg({'count':'sum', 'count_patient':'sum'}).reset_index()

market_scan_ur = market_scan_urban.merge(market_scan_rural, on=['code', 'year'], suffixes=('_urban', '_rural'), how='inner')
code_set = set(market_scan_urban['code']).intersection(set(market_scan_rural['code']))

# total
odds_ratio_df = []
for code, sub_df in market_scan_ur.groupby('code'):
    n_u_pos = int(sub_df['count_urban'].mean())
    n_u_neg = int(sub_df['count_patient_urban'].mean()) - n_u_pos
    n_r_pos = int(sub_df['count_rural'].mean())
    n_r_neg = int(sub_df['count_patient_rural'].mean()) - n_r_pos
    odds_ratio_result = odds_ratio([[n_u_pos, n_u_neg], [n_r_pos, n_r_neg]], kind='sample')
    odds_ratio_ci = odds_ratio_result.confidence_interval(0.95)
    stat, p_val = fisher_exact([[n_u_pos, n_u_neg], [n_r_pos, n_r_neg]], alternative='two-sided')
    odds_ratio_df.append({'code':code, 'odds_ratio':odds_ratio_result.statistic, 'ci_low':odds_ratio_ci[0], 'ci_high':odds_ratio_ci[1],
                          'fisher_exact_stat':stat, 'fisher_exact_pval':p_val})
odds_ratio_df = pd.DataFrame(odds_ratio_df)
odds_ratio_df['description'] = odds_ratio_df['code'].apply(lambda x: icd10[x]['description'])
odds_ratio_df['odds_ratio_str'] = odds_ratio_df['odds_ratio'].apply(lambda x: f'{x:.5f}')
odds_ratio_df['ci_low_str'] = odds_ratio_df['ci_low'].apply(lambda x: f'{x:.5f}')
odds_ratio_df['ci_high_str'] = odds_ratio_df['ci_high'].apply(lambda x: f'{x:.5f}')
odds_ratio_df['fisher_exact_stat_str'] = odds_ratio_df['fisher_exact_stat'].apply(lambda x: f'{x:.5f}')
odds_ratio_df['fisher_exact_pval_str'] = odds_ratio_df['fisher_exact_pval'].apply(lambda x: f'{x:.5f}')
odds_ratio_df_ = odds_ratio_df[['code', 'odds_ratio_str', 'ci_low_str', 'ci_high_str', 'fisher_exact_stat_str', 
                                'fisher_exact_pval_str', 'description', 'odds_ratio', 'ci_low', 'ci_high', 
                                'fisher_exact_stat', 'fisher_exact_pval']].sort_values(by=['fisher_exact_pval','odds_ratio'])
odds_ratio_df_.to_csv(os.path.join(result_dir, f'ru_odds_ratio_{icd_level}.csv'), index=False)