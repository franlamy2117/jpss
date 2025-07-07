import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def day_hour_heatmap(df_ref, title):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df = df_ref.copy()
    df2 = pd.DataFrame()
    df2['datetime'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    df2['created_at'] = df2['datetime'].dt.tz_convert('Asia/Bangkok')
    df2['hour'] = df2['created_at'].dt.hour
    df2['weekday'] = df2['created_at'].dt.day_name()
    df2['weekday'] = pd.Categorical(df2.weekday, categories=days, ordered=True)
    df3 = df2.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    x_axis_labels = ['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00',
                     '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00',
                     '23:00']
    df3.to_csv(rf"D:\CAS tweets\JPSS revision\Final\timeperday_{title}.csv")
    sns.heatmap(df3, annot=False, cbar=True, xticklabels=x_axis_labels, yticklabels=days)
    plt.show()
