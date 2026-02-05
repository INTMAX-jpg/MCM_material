import pandas as pd

# Load data
df = pd.read_csv('C_origin.csv')

# Extract unique countries
countries = df['celebrity_homecountry/region'].unique()

# Defined mapping in 1_dim_characterastic.py
country_to_continent = {
    'United States': 'North America',
    'New Zealand': 'Oceania',
    'England': 'Europe',
    'Mexico': 'North America',
    'Canada': 'North America',
    'Czechoslovakia': 'Europe',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Yugoslavia': 'Europe',
    'France': 'Europe',
    'Australia': 'Oceania',
    'Russia': 'Europe', 
    'Ukraine': 'Europe',
    'Poland': 'Europe',
    'Italy': 'Europe',
    'Ireland': 'Europe',
    'Cuba': 'North America',
    'Israel': 'Asia',
    'Philippines': 'Asia',
    'India': 'Asia',
    'South Africa': 'Africa',
    'Germany': 'Europe',
    'Sweden': 'Europe',
    'Latvia': 'Europe',
    'Albania': 'Europe',
    'Slovenia': 'Europe',
    'Colombia': 'South America',
    'Argentina': 'South America',
    'Panama': 'North America',
    'Puerto Rico': 'North America'
}

print("Countries mapped to 'Others':")
others = []
for c in countries:
    if pd.isna(c):
        continue
    c_str = str(c).strip()
    if c_str not in country_to_continent:
        others.append(c_str)

for o in sorted(others):
    print(o)
