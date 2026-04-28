# Rename columns to standard names
df.rename(columns={
    'Engine_Temp': 'Temperature',
    'Temp': 'Temperature',
    'temperature': 'Temperature'
}, inplace=True)