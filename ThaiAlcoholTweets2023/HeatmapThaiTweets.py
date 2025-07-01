import pandas as pd
import calplot as cal
import matplotlib.pyplot as plt
import seaborn as sns


def heat_map_date():
    df = pd.read_csv(r'C:\Users\Francois\OneDrive\Bureau\Grants\CAS\CAS tweets\time neg.csv',
                     delimiter=',')
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    # df['date2'] = df['date']
    # df.groupby('date2').size().rename('count')
    df.set_index('date', inplace=True)
    print(df.info())
    cal.calplot(df['count'], how='sum')


def heat_map_hour():
    df = pd.read_csv(r'C:\Users\Francois\OneDrive\Bureau\labelled data negative curated seaborn.csv',
                     parse_dates=['created_at'])
    df['created_at'].dt.tz_convert('Asia/Vientiane')
    df['weekday'] = df['created_at'].dt.dayofweek
    df['hour'] = df['created_at'].dt.hour
    df = df.pivot_table(index=df['weekday'], columns=df['hour'], aggfunc='sum')
    x_axis_labels = ['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00',
                     '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00',
                     '23:00']
    y_axis_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(df)
    sns.heatmap(df, annot=False, cbar=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.show()


# heat_map_date()
heat_map_hour()
