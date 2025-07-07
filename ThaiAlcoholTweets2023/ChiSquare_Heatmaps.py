import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import numpy as np

def chi_square_by_hour(pc_net, pc_pos, pc_neg):
    df_ref =  pc_net.copy()
    df_sub1 = pc_pos.copy()
    df_sub2 = pc_neg.copy()
    df2 = pd.DataFrame()
    df2['datetime'] = pd.to_datetime(df_ref['created_at'], utc=True, errors='coerce')
    df2['datetime'] = df2['datetime'].dt.tz_convert("Asia/Bangkok")
    df3 = pd.DataFrame()
    df3['datetime'] = pd.to_datetime(df_sub1['created_at'], utc=True, errors='coerce')
    df3['datetime'] = df3['datetime'].dt.tz_convert("Asia/Bangkok")
    df4 = pd.DataFrame()
    df4['datetime'] = pd.to_datetime(df_sub2['created_at'], utc=True, errors='coerce')
    df4['datetime'] = df4['datetime'].dt.tz_convert("Asia/Bangkok")
    ref_counts = df2['datetime'].dt.strftime('%w-%H').value_counts().sort_index()
    sub_counts = df3['datetime'].dt.strftime('%w-%H').value_counts().reindex(ref_counts.index, fill_value=0)
    sub_counts2 = df4['datetime'].dt.strftime('%w-%H').value_counts().reindex(ref_counts.index, fill_value=0)
    contingency = pd.concat([ref_counts, sub_counts, sub_counts2], axis=1).fillna(0)
    chi2, p, dof, expected = chi2_contingency(contingency.T)
    print(f"\nChi-square test for time distribution:")
    print(f"Chi² = {chi2:.2f}, df = {dof}, p = {p:.5f}")
    n = contingency.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    cramer_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    print(f"Cramer's V = {cramer_v:.4f}")
    return contingency
