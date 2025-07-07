import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import numpy as np

def chi_square_by_hour(df, df2):
    df_ref = pd.read_csv(df)
    df_subset = pd.read_csv(df2)
    df2 = pd.DataFrame()
    df2['datetime'] = pd.to_datetime(df_ref['created_at'], utc=True, errors='coerce')
    df2['datetime'] = df2['datetime'].dt.tz_convert("Asia/Bangkok")
    df3 = pd.DataFrame()
    df3['datetime'] = pd.to_datetime(df_subset['created_at'], utc=True, errors='coerce')
    df3['datetime'] = df3['datetime'].dt.tz_convert("Asia/Bangkok")
    ref_counts = df2['datetime'].dt.strftime('%w-%H').value_counts().sort_index()
    sub_counts = df3['datetime'].dt.strftime('%w-%H').value_counts().reindex(ref_counts.index, fill_value=0)
    contingency = pd.concat([ref_counts, sub_counts], axis=1, keys=['All_PC', 'Subset']).fillna(0)
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

def kruskal_wallis_by_hour(path):
    df = pd.read_csv(path)
    h_statistic, p_value = kruskal(df['Positive'], df['Negative'], df['Neutral'])
    print(f"Kruskal_Wallis H-statistic: {h_statistic}")
    print(f"Kruskal_Wallis P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant differences found.")

def chi_square_by_hour2(path1):
    df_ref = pd.read_csv(path1)
    ref_counts = df_ref['Neutral']
    sub_counts = df_ref['Negative']
    p_counts = df_ref['Positive']
    contingency = pd.concat([ref_counts, sub_counts, p_counts], axis=1).fillna(0)
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


# chi_square_by_hour(r'D:\CAS tweets\JPSS revision\Final\Neutral_sentiments.csv',
#                    r'D:\CAS tweets\JPSS revision\Final\Negative_sentiments.csv')
chi_square_by_hour2(r"D:\CAS tweets\JPSS revision\Final\chisquaredata.csv")
kruskal_wallis_by_hour(r"D:\CAS tweets\JPSS revision\Final\chisquaredata.csv")
