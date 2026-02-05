import pandas as pd
import numpy as np
import os

# ==========================================
# Configuration
# ==========================================
input_file = 'C_origin.csv'
if not os.path.exists(input_file):
    # Try parent directory if not found in current
    if os.path.exists(f'../{input_file}'):
        input_file = f'../{input_file}'
        
output_file = 'cur代码/四个平均值.csv'
season_start = 1
season_final = 34

# ==========================================
# Classification Logic (Copied from 1_dim_characterastic.py)
# ==========================================

# 1. Industry
target_industries = [
    'Actor/Actress', 
    'Athlete', 
    'Comedian', 
    'Model', 
    'Singer/Rapper', 
    'TV Personality'
]

def map_industry(ind):
    if not isinstance(ind, str):
        return 'Others'
    ind = ind.strip()
    if ind in target_industries:
        return ind
    return 'Others'

# 2. Region
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
    'Puerto Rico': 'North America',
    'Croatia': 'Europe',
    'South Korea': 'Asia',
    'Spain': 'Europe',
    'Taiwan China': 'Asia',
    'Venezuela': 'South America',
    'Wales': 'Europe'
}

def map_country(country):
    if not isinstance(country, str):
        return 'Others'
    country = country.strip()
    return country_to_continent.get(country, 'Others')

# 3. Age
def map_age_group(age):
    try:
        age = int(age)
        lower = (age // 10) * 10
        upper = lower + 9
        return f"{lower}-{upper}"
    except:
        return "Unknown"

# ==========================================
# Main Execution
# ==========================================
def main():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Read data
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Filter Season
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(subset=['season'])
    df = df[(df['season'] >= season_start) & (df['season'] <= season_final)]
    
    # Normalize Columns
    rename_map = {
        'celebrity_homecountry_region': 'celebrity_homecountry/region',
        'celebrity_homecountry': 'celebrity_homecountry/region',
        'age_during_season': 'celebrity_age_during_season'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure Placement is numeric
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
    
    # Drop rows with NaN placement for calculation
    df = df.dropna(subset=['placement'])
    
    results = []

    # ----------------------------------
    # 1. Industry
    # ----------------------------------
    if 'celebrity_industry' in df.columns:
        df['industry_group'] = df['celebrity_industry'].apply(map_industry)
        means = df.groupby('industry_group')['placement'].mean()
        for cls, val in means.items():
            results.append({
                'character': 'industry',
                'character_class': cls,
                'mean_placement': val
            })
            
    # ----------------------------------
    # 2. Region (homeregion)
    # ----------------------------------
    if 'celebrity_homecountry/region' in df.columns:
        df['region_group'] = df['celebrity_homecountry/region'].apply(map_country)
        means = df.groupby('region_group')['placement'].mean()
        for cls, val in means.items():
            results.append({
                'character': 'homeregion',
                'character_class': cls,
                'mean_placement': val
            })
            
    # ----------------------------------
    # 3. Age
    # ----------------------------------
    if 'celebrity_age_during_season' in df.columns:
        # Ensure age is numeric first
        df['age_num'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
        # Filter out invalid ages for grouping
        df_age = df.dropna(subset=['age_num']).copy()
        df_age['age_group'] = df_age['age_num'].apply(map_age_group)
        means = df_age.groupby('age_group')['placement'].mean()
        for cls, val in means.items():
            results.append({
                'character': 'age',
                'character_class': cls,
                'mean_placement': val
            })
            
    # ----------------------------------
    # 4. Ballroom Partner
    # ----------------------------------
    if 'ballroom_partner' in df.columns:
        # No special classification needed, use raw values
        means = df.groupby('ballroom_partner')['placement'].mean()
        for cls, val in means.items():
            results.append({
                'character': 'ballroom_partner',
                'character_class': cls,
                'mean_placement': val
            })
            
    # Save results
    res_df = pd.DataFrame(results)
    
    # Reorder columns just in case
    res_df = res_df[['character', 'character_class', 'mean_placement']]
    
    res_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
