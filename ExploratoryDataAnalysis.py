import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('terroism.csv', encoding='latin1', low_memory=False)
print(df.head())
print(df.info())
print(df.describe())
plt.figure(figsize=(12, 6))
sns.countplot(x='attacktype1_txt', data=df)
plt.xticks(rotation=90)
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.title('Distribution of Attack Types')
plt.show()
df['iyear'] = pd.to_datetime(df['iyear'], format='%Y')
plt.figure(figsize=(12, 6))
df['iyear'].dt.year.value_counts().sort_index().plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.title('Terrorist Incidents Over Time')
plt.show()
country_counts = df['country_txt'].value_counts()
top_countries = country_counts.head(20).index.tolist()
top_country_data = df[df['country_txt'].isin(top_countries)]
pivot_table = top_country_data.pivot_table(index='iyear', columns='country_txt', values='eventid', aggfunc='count', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
plt.xlabel('Country')
plt.ylabel('Year')
plt.title('Terrorist Incidents by Country (Top 20)')
plt.show()
avg_fatalities_by_type = df.groupby('attacktype1_txt')['nkill'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
avg_fatalities_by_type.plot(kind='bar')
plt.xlabel('Attack Type')
plt.ylabel('Average Fatalities')
plt.title('Average Fatalities by Attack Type')
plt.show()