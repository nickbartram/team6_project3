# import dependencies
import psycopg2
import pandas as pd
from keys import post_username, post_password
from sqlalchemy import create_engine

# Set up database connection parameters
conn = psycopg2.connect(
    dbname = "team6_project3_db",
    user = post_username,
    password = post_password,
    host = "database-1.croamw4iqxpi.us-east-2.rds.amazonaws.com",
    port = "5432" 
)

# Create database query
query = "SELECT * FROM climate_impact_agriculture;"

# Create the pandas database
data = pd.read_sql_query(query, conn)

# Rename the headers back to their original state, with capitalization
data.rename(columns={
    'year': 'Year',
    'country': 'Country',
    'region': 'Region',
    'crop_type': 'Crop_Type',
    'average_temperature_c': 'Average_Temperature_C',
    'total_precipitation_mm': 'Total_Precipitation_mm',
    'co2_emissions_mt': 'CO2_Emissions_MT',
    'crop_yield_mt_per_ha': 'Crop_Yield_MT_per_HA',
    'extreme_weather_events': 'Extreme_Weather_Events',
    'pesticide_use_kg_per_ha': 'Pesticide_Use_KG_per_HA',
    'fertilizer_use_kg_per_ha': 'Fertilizer_Use_KG_per_HA',
    'soil_health_index': 'Soil_Health_Index',
    'adaptation_strategies': 'Adaptation_Strategies',
    'economic_impact_million_usd': 'Economic_Impact_Million_USD'
}, inplace=True)

# Create a df using pandas to read the CSV of owid data (Our World in Data)
owid_df = pd.read_csv('Resources\owid-co2-data.csv')

# Create variables for the engine connection
dbname = "team6_project3_db"
host = "database-1.croamw4iqxpi.us-east-2.rds.amazonaws.com"
port = "5432" 

# Create a connection to the database
engine = create_engine(f'postgresql://{post_username}:{post_password}@{host}:{port}/{dbname}')

# Import the DataFrame to the PostgreSQL database using the established connection
owid_df.to_sql('owid_co2_data', engine, if_exists='append', index=False)