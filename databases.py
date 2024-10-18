# import dependencies
import pandas as pd
from keys import post_username, post_password
from sqlalchemy import create_engine
import pycountry

# Global variables
cleaned_df = None
data = None

def process_owid_data():
    global cleaned_df
    
    # Set up database connection parameters
    dbname = "team6_project3_db"
    host = "database-1.croamw4iqxpi.us-east-2.rds.amazonaws.com"
    port = "5432"

    # Create the SQLAlchemy engine
    engine = create_engine(f'postgresql://{post_username}:{post_password}@{host}:{port}/{dbname}')

    # Create database query
    query = "SELECT * FROM owid_co2_data;"

    # Create the pandas database
    owid_df = pd.read_sql_query(query, engine)
    
    # Create a new df with only the relevant columns
    reduced_df = owid_df[['country', 'year', 'co2']]

    # There are a lot of NaN values, I could try to eliminate those
    reduced_df = reduced_df.dropna(subset=['co2'])

    # Create a set of valid countries from the pycountry library
    valid_countries = {country.name for country in pycountry.countries}

    cleaned_df = reduced_df[reduced_df['country'].isin(valid_countries)].copy()

def fetch_climate_impact_data():
    global data
    
    # Set up database connection parameters
    dbname = "team6_project3_db"
    host = "database-1.croamw4iqxpi.us-east-2.rds.amazonaws.com"
    port = "5432"

    # Create the SQLAlchemy engine
    engine = create_engine(f'postgresql://{post_username}:{post_password}@{host}:{port}/{dbname}')

    # Create database query
    query = "SELECT * FROM climate_impact_agriculture;"

    # Create the pandas database
    data = pd.read_sql_query(query, engine)

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

# Run the data processing functions when the module is imported
process_owid_data()
fetch_climate_impact_data()