import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set page layout to wide and adjust sidebar and alignment
st.set_page_config(layout="wide")

# Inject custom CSS to left-align the layout
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 85%;  /* Adjust width */
        padding-left: 10px;
        padding-right: 10px;
        text-align: left; /* Align text to the left */
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
data = pd.read_csv(r'C:\Users\Shari\Desktop\Data Science Course\Challenges\team6_project3\Resources\climate_change_impact_on_agriculture_2024.csv')

# Streamlit app structure
st.title("Climate Change Impact on Agriculture Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox("Select Country", options=data['Country'].unique())
filtered_data = data[data['Country'] == selected_country]

selected_region = st.sidebar.selectbox("Select Region", options=filtered_data['Region'].unique())
filtered_data = filtered_data[filtered_data['Region'] == selected_region]


#Step 1 Aggregate mean values of continuous variables over time. Focus on avg temp and crop yield
mean_data_avgTemp_cropYield = filtered_data.groupby('Year').mean(numeric_only=True)[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']]
movingAvg_avgTemp_cropYield = mean_data_avgTemp_cropYield.rolling(window=5).mean()




#ADD ALL REGION AND ALL COUNTRY DEFAULTS







# Add a Crop Type filter with "All Crops" option
crop_type_options = ['All Crops'] + list(filtered_data['Crop_Type'].unique())
selected_crop_type = st.sidebar.selectbox("Select Crop Type", options=crop_type_options)

# Apply Crop Type filter (skip filtering if "All Crops" is selected)
if selected_crop_type != 'All Crops':
    filtered_data = filtered_data[filtered_data['Crop_Type'] == selected_crop_type]

# Year range filter
year_range = st.sidebar.slider("Select Year Range", int(data['Year'].min()), int(data['Year'].max()), (2000, 2024))
filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]


# Interactive Plot 1: Temperature vs Crop Yield

# Filter the data to focus on relevant columns for emissions
filtered_ghg_data = data[['Year', 'CO2_Emissions_MT']].dropna()

# Group by year and calculate the mean emissions to smooth out the line
filtered_ghg_data = filtered_ghg_data.groupby('Year')['CO2_Emissions_MT'].mean().reset_index()

# Ensure the data is sorted by year
filtered_ghg_data = filtered_ghg_data.sort_values(by='Year')

# Prepare data for polynomial regression model
x = filtered_ghg_data['Year'].values.reshape(-1, 1)
y = filtered_ghg_data['CO2_Emissions_MT'].values

# Use polynomial features to capture non-linear trends
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Train polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)

# Predict future emissions (up to 2030 for projection)
future_years = np.arange(data['Year'].min(), 2031).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predicted_emissions = model.predict(future_years_poly)

# Calculate the goal (20% reduction from the predicted 2030 emissions)
goal_2030 = predicted_emissions[-1] * 0.8

# Create goal line (linear reduction from current to 2030 goal)
current_year = filtered_ghg_data['Year'].max()
current_emissions = filtered_ghg_data.loc[filtered_ghg_data['Year'] == current_year, 'CO2_Emissions_MT'].values[0]
goal_years = np.arange(current_year, 2031)
goal_emissions = np.linspace(current_emissions, goal_2030, len(goal_years))

# Create a new figure for GHG emissions
fig1 = go.Figure()

# Add actual emissions line (smoothed)
fig1.add_trace(go.Scatter(
    x=filtered_ghg_data['Year'],
    y=filtered_ghg_data['CO2_Emissions_MT'],
    mode='lines',
    name='Actual CO2 Emissions',
    line=dict(color='red', width=2)
))

# Add predicted emissions line (polynomial regression)
fig1.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predicted_emissions,
    mode='lines',
    name='Predicted CO2 Emissions',
    line=dict(color='orange', width=2, dash='dash')
))

# Add goal line
fig1.add_trace(go.Scatter(
    x=goal_years,
    y=goal_emissions,
    mode='lines',
    name='Emissions Reduction Goal',
    line=dict(color='green', width=2, dash='dot')
))

# Determine if projections are on track
final_predicted = predicted_emissions[-1]
final_goal = goal_emissions[-1]
on_track = final_predicted <= final_goal

# Add status indicator
status_color = 'green' if on_track else 'red'
status_text = 'On Track' if on_track else 'Off Track'

fig1.add_annotation(
    x=0.98,
    y=0.98,
    xref="paper",
    yref="paper",
    text=f"Status: {status_text}",
    showarrow=False,
    font=dict(size=16, color="white"),
    align="center",
    bordercolor=status_color,
    borderwidth=2,
    borderpad=4,
    bgcolor=status_color,
    opacity=0.8
)

# Update layout for the emissions chart
fig1.update_layout(
    title='GHG Emissions: Actual, Predicted, and Reduction Goal',
    xaxis_title='Year',
    yaxis_title='CO2 Emissions (Million Tons)',
    plot_bgcolor='white',
    hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    showlegend=True
)

# Display the GHG emissions chart
st.plotly_chart(fig1)

 
# Interactive Plot 2: Precipitation and CO2 Emissions Over Time
st.subheader("Precipitation and CO2 Emissions Over Time")
    # Calculate the moving average (5-year window)


# Create traces for Average Temperature and Crop Yield
fig2 = go.Figure()  #FIX GRIDLINESS

# Plot Average Temperature
fig2.add_trace(go.Scatter(
    x=mean_data_avgTemp_cropYield.index,
    y=mean_data_avgTemp_cropYield['Average_Temperature_C'],
    mode='lines',
    name='Average Temperature (C)',
    line=dict(color='blue', width=0.7),
    opacity=0.3
))

# Plot Moving Average of Average Temperature
fig2.add_trace(go.Scatter(
    x=movingAvg_avgTemp_cropYield.index,
    y=movingAvg_avgTemp_cropYield['Average_Temperature_C'],
    mode='lines',
    name='Average Temperature (C, moving 5y)',
    line=dict(color='darkblue', width=1.5, dash='dash')
))

# Plot Crop Yield on a secondary y-axis
fig2.add_trace(go.Scatter(
    x=mean_data_avgTemp_cropYield.index,
    y=mean_data_avgTemp_cropYield['Crop_Yield_MT_per_HA'],
    mode='lines',
    name='Crop Yield (MT/HA)',
    line=dict(color='green', width=0.7),
    opacity=0.3,
    yaxis='y2'
))

# Plot Moving Average of Crop Yield
fig2.add_trace(go.Scatter(
    x=movingAvg_avgTemp_cropYield.index,
    y=movingAvg_avgTemp_cropYield['Crop_Yield_MT_per_HA'],
    mode='lines',
    name='Crop Yield (MT/HA, moving 5y)',
    line=dict(color='darkgreen', width=1.5, dash='dash'),
    yaxis='y2'
))

# Update layout for dual y-axes
fig2.update_layout(
    title='Average Temperature and Crop Yield Over Time',
    xaxis=dict(title='Year'),
    yaxis=dict(
        title='Average Temperature (C)',
        titlefont=dict(color='white'),
        tickfont=dict(color='white')
    ),
    yaxis2=dict(
        title='Crop Yield (MT/HA)',
        titlefont=dict(color='white'),
        tickfont=dict(color='white'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'),
    margin=dict(l=50, r=50, t=50, b=50),
    hovermode='x',
    plot_bgcolor='white'
)

# Add gridlines and remove spines (axes borders)
fig2.update_xaxes(showgrid=True, gridcolor='lightgray')
fig2.update_yaxes(showgrid=True, gridcolor='lightgray')
fig2.update_yaxes(showline=False)  # Disable spines

# Display chart with filters applied
st.plotly_chart(fig2)

# Interactive Plot 3: Economic Impact by Region  #ADD ANNOTATIONS to columns
st.subheader("Economic Impact by Region")
fig3 = px.bar(filtered_data, x='Region', y='Economic_Impact_Million_USD', color='Crop_Type',
              title="Economic Impact by Region and Crop Type")
st.plotly_chart(fig3)

# Interactive Plot 4: Extreme Weather Events vs Crop Yield
st.subheader("Extreme Weather Events vs Crop Yield")
fig4 = px.scatter(filtered_data, x='Extreme_Weather_Events', y='Crop_Yield_MT_per_HA', 
                  size='Total_Precipitation_mm', color='Crop_Type', hover_name='Region',
                  title="Extreme Weather Events vs Crop Yield")
st.plotly_chart(fig4)

# Adding the Animated Time Series Chart
st.subheader("Animated Time Series: Crop Yield Over Time")
filtered_data = filtered_data.sort_values(by='Year')  # Sort data by year for proper chronological order

fig5 = px.scatter(filtered_data, 
                  x='Average_Temperature_C', 
                  y='Crop_Yield_MT_per_HA', 
                  size='Extreme_Weather_Events', 
                  color='Crop_Type', 
                  hover_name='Region', 
                  animation_frame='Year', 
                  animation_group='Region', 
                  range_x=[filtered_data['Average_Temperature_C'].min(), filtered_data['Average_Temperature_C'].max()],
                  range_y=[filtered_data['Crop_Yield_MT_per_HA'].min(), filtered_data['Crop_Yield_MT_per_HA'].max()],
                  title="Crop Yield vs Temperature Over Time (Animated)")

# Setting axis limits to ensure proper time filtering and smoother transitions
fig5.update_layout(xaxis_title='Average Temperature (°C)', yaxis_title='Crop Yield (MT per HA)')
st.plotly_chart(fig5)

# Displaying filter details
st.sidebar.write("Filters applied: Country - {}, Region - {}, Crop Type - {}, Year Range - {}".format(selected_country, selected_region, selected_crop_type, year_range))



