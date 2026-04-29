import pandas as pd
import sqlite3
import os

csv_file = 'PSX_Oil_Sector_Combined_2016_2026(Sheet1).csv'
db_file = 'psx_oil_data.db'

print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file)

# Clean the data before inserting it into the database
def parse_volume(x):
    if pd.isna(x):
        return 0
    x = str(x).replace(',', '')
    if 'K' in x:
        return float(x.replace('K', '')) * 1e3
    elif 'M' in x:
        return float(x.replace('M', '')) * 1e6
    elif 'B' in x:
        return float(x.replace('B', '')) * 1e9
    else:
        try:
            return float(x)
        except ValueError:
            return 0

df['Volume'] = df['Vol.'].apply(parse_volume)

# Ensure numerical columns are correctly formatted
for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# Write to SQLite Database
print(f"Connecting to {db_file}...")
conn = sqlite3.connect(db_file)

print("Writing to database...")
df.to_sql('stock_data', conn, if_exists='replace', index=False)

# Create an index on Symbol and Date for faster querying
cursor = conn.cursor()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_data (Symbol, Date)")
conn.commit()

conn.close()
print("✅ Database created successfully!")
