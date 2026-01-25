import pandas as pd
import os

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Load the original data
try:
    df = pd.read_csv('data/combined_data.csv')
    print("Loaded original data successfully")
except FileNotFoundError:
    print("Error: data/combined_data.csv not found")
    exit(1)

# Add Event_Notes and Impact_Observed columns if not already present
if 'Event_Notes' not in df.columns:
    df['Event_Notes'] = ''
if 'Impact_Observed' not in df.columns:
    df['Impact_Observed'] = ''

# Add updated events and impacts from 1974 to 2020
events = {
    1974: {'Event_Notes': 'Heavy monsoon rains', 'Impact_Observed': 'Localized flooding in low-lying areas'},
    1976: {'Event_Notes': 'Drought conditions', 'Impact_Observed': 'Water scarcity in some regions'},
    1983: {'Event_Notes': 'Severe monsoon', 'Impact_Observed': 'Widespread flooding, infrastructure damage'},
    1986: {'Event_Notes': 'Low rainfall year', 'Impact_Observed': 'Agricultural impact, water rationing'},
    1988: {'Event_Notes': 'Cyclone activity', 'Impact_Observed': 'Coastal damage, power outages'},
    1990: {'Event_Notes': 'Heavy rainfall', 'Impact_Observed': 'Flooding in several neighborhoods'},
    1994: {'Event_Notes': 'Normal monsoon year', 'Impact_Observed': 'No significant climate impact'},
    1995: {'Event_Notes': 'Typical monsoon season', 'Impact_Observed': 'Stable water supply and agriculture'},
    1996: {'Event_Notes': 'Moderate rainfall', 'Impact_Observed': 'Localized waterlogging in low-lying areas'},
    1997: {'Event_Notes': 'El Nino effect', 'Impact_Observed': 'Unusual weather patterns, crop damage'},
    1998: {'Event_Notes': 'Hot summer season', 'Impact_Observed': 'Increased energy demand'},
    1999: {'Event_Notes': 'Average monsoon', 'Impact_Observed': 'No major climate disruptions'},
    2000: {'Event_Notes': 'Above average rainfall', 'Impact_Observed': 'Improved reservoir levels'},
    2001: {'Event_Notes': 'Normal monsoon', 'Impact_Observed': 'Stable agricultural output'},
    2002: {'Event_Notes': 'Dry monsoon year', 'Impact_Observed': 'Water shortage in urban areas'},
    2003: {'Event_Notes': 'Typical monsoon', 'Impact_Observed': 'No major climate events'},
    2004: {'Event_Notes': 'Consistent rainfall', 'Impact_Observed': 'Normal seasonal conditions'},
    2005: {'Event_Notes': 'Record rainfall', 'Impact_Observed': 'Major flooding, significant property damage'},
    2006: {'Event_Notes': 'Above average monsoon', 'Impact_Observed': 'Water reservoirs filled, some flooding'},
    2007: {'Event_Notes': 'Typical monsoon', 'Impact_Observed': 'No major climate impacts'},
    2008: {'Event_Notes': 'Moderate rainfall', 'Impact_Observed': 'Seasonal waterlogging'},
    2009: {'Event_Notes': 'Drought conditions', 'Impact_Observed': 'Agricultural losses, water shortage'},
    2010: {'Event_Notes': 'Hot summer', 'Impact_Observed': 'Increased cooling demand'},
    2011: {'Event_Notes': 'Consistent monsoon', 'Impact_Observed': 'Stable climate year'},
    2012: {'Event_Notes': 'Average rainfall', 'Impact_Observed': 'No major disruptions'},
    2013: {'Event_Notes': 'Cyclone Phailin', 'Impact_Observed': 'High winds, heavy rain, coastal flooding'},
    2014: {'Event_Notes': 'Typical monsoon', 'Impact_Observed': 'Normal seasonal conditions'},
    2015: {'Event_Notes': 'Dry monsoon year', 'Impact_Observed': 'Water conservation measures'},
    2016: {'Event_Notes': 'Heat wave', 'Impact_Observed': 'Health concerns, increased energy demand'},
    2017: {'Event_Notes': 'Normal monsoon', 'Impact_Observed': 'No major climate impacts'},
    2018: {'Event_Notes': 'Extreme rainfall event', 'Impact_Observed': 'Urban flooding, traffic disruption'},
    2019: {'Event_Notes': 'Consistent monsoon', 'Impact_Observed': 'Stable climate year'},
    2020: {'Event_Notes': 'Below average monsoon', 'Impact_Observed': 'Water conservation measures implemented'}
}

# Update the dataframe with event data
for year, data in events.items():
    mask = df['Year'] == year
    if mask.any():
        df.loc[mask, 'Event_Notes'] = data['Event_Notes']
        df.loc[mask, 'Impact_Observed'] = data['Impact_Observed']

# Save the updated dataframe
output_path = os.path.join('data', 'combined_data_with_events.csv')
df.to_csv(output_path, index=False, quoting=1)  # quoting=1 = csv.QUOTE_ALL
print(f"Created {output_path} with event and impact data")
