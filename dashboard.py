import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Load data
from databases import cleaned_df, data



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



# Streamlit app structure
st.title("Climate Change Impact on Agriculture Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Add a Country filter option
country_options = sorted(data['Country'].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", options=country_options, index=0)

# Initialize filtered_data with the full dataset
filtered_data = data[data['Country'] == selected_country]


# Add a Region filter with "All Regions" option
region_options = ['All Regions'] + sorted(filtered_data['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", options=region_options)

# Apply Region filter
if selected_region != 'All Regions':
    filtered_data = filtered_data[filtered_data['Region'] == selected_region]

# Add a Crop Type filter with "All Crops" option
crop_type_options = ['All Crops'] + sorted(filtered_data['Crop_Type'].unique().tolist())
selected_crop_type = st.sidebar.selectbox("Select Crop Type", options=crop_type_options)

# Apply Crop Type filter
if selected_crop_type != 'All Crops':
    filtered_data = filtered_data[filtered_data['Crop_Type'] == selected_crop_type]

# Year range filter
min_year = int(data['Year'].min())
max_year = int(data['Year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Apply Year range filter
filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]


# Interactive Plot 1: Temperature vs Crop Yield

# Filter the data to focus on relevant columns for emissions
filtered_ghg_data = data[['Year', 'CO2_Emissions_MT']].dropna()

# Group by year and calculate the mean emissions to smooth out the line
filtered_ghg_data = filtered_ghg_data.groupby('Year')['CO2_Emissions_MT'].mean().reset_index()

# Ensure the data is sorted by year
filtered_ghg_data = filtered_ghg_data.sort_values(by='Year')
# Calculate average CO2 emissions per year across all countries
average_emissions = cleaned_df.groupby('year')['co2'].mean().reset_index()
average_emissions = average_emissions.rename(columns={'year': 'Year', 'co2': 'CO2_Emissions_MT'})

# Prepare data for polynomial regression model
x = average_emissions['Year'].values.reshape(-1, 1)
y = average_emissions['CO2_Emissions_MT'].values

# Use polynomial features to capture non-linear trends
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Train polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)

# Predict future emissions (up to 2030 for projection)
future_years = np.arange(average_emissions['Year'].min(), 2031).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predicted_emissions = model.predict(future_years_poly)

# Calculate the goal (5% reduction from the predicted 2030 emissions)
goal_2030 = predicted_emissions[-1] * 0.95

# Create goal line (linear reduction from current to 2030 goal)
current_year = average_emissions['Year'].max()
current_emissions = average_emissions.loc[average_emissions['Year'] == current_year, 'CO2_Emissions_MT'].values[0]
goal_years = np.arange(current_year, 2031)
goal_emissions = np.linspace(current_emissions, goal_2030, len(goal_years))

# Create a new figure for GHG emissions
fig1 = go.Figure()

# Add actual emissions line (average across countries)
fig1.add_trace(go.Scatter(
    x=average_emissions['Year'],
    y=average_emissions['CO2_Emissions_MT'],
    mode='lines',
    name='Actual CO2 Emissions (Average)',
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
    x=0.02,
    y=0.98,
    xref="paper",
    yref="paper",
    text=f"Status: {status_text}",
    showarrow=False,
    font=dict(size=16, color="white"),
    align="left",
    bordercolor=status_color,
    borderwidth=2,
    borderpad=4,
    bgcolor=status_color,
    opacity=0.8
)

# Update layout for the emissions chart
fig1.update_layout(
    title='Average Global GHG Emissions: Actual, Predicted, and Reduction Goal',
    xaxis_title='Year',
    yaxis_title='Average CO2 Emissions (Million Tons)',
    plot_bgcolor='white',
    hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    showlegend=True
)

# Display the GHG emissions chart
st.plotly_chart(fig1)

 
# Interactive Plot 2: Precipitation and CO2 Emissions Over Time
#Step 1 Aggregate mean values of continuous variables over time. Focus on avg temp and crop yield
mean_data_avgTemp_cropYield = filtered_data.groupby('Year').mean(numeric_only=True)[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']]
movingAvg_avgTemp_cropYield = mean_data_avgTemp_cropYield.rolling(window=5).mean()
st.subheader("Precipitation and CO2 Emissions Over Time")
    # Calculate the moving average (5-year window)


# Create traces for Average Temperature and Crop Yield
fig2 = go.Figure()  

# Plot Average Temperature
fig2.add_trace(go.Scatter(
    x=mean_data_avgTemp_cropYield.index,
    y=mean_data_avgTemp_cropYield['Average_Temperature_C'],
    mode='lines',
    name='Average Temperature (C)',
    line=dict(color='blue', width=0.7),
    opacity=0.6
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
    opacity=0.6,
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
        tickfont=dict(color='white'),
        gridcolor='rgba(200, 200, 200, 0.4)',  # Light gray with 20% opacity
    ),
    yaxis2=dict(
        title='Crop Yield (MT/HA)',
        titlefont=dict(color='white'),
        tickfont=dict(color='white'),
        overlaying='y',
        side='right',
        showgrid=False,
        zeroline=False
    ),
    legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'),
    margin=dict(l=50, r=50, t=50, b=50),
    hovermode='x',
    plot_bgcolor='white'
)

# Display chart with filters applied
st.plotly_chart(fig2)


# Interactive Plot 6: Extreme Weather Event Correlation Matrix
# Correlation analysis between extreme weather events and economic impact by country and region
corr_country_region = data.groupby(['Country', 'Region']).apply(lambda group: group['Extreme_Weather_Events'].corr(group['Economic_Impact_Million_USD'])).reset_index(name='Correlation')
# Create pivot table, country as rows and regions as columns
pivot_corr_country_region = corr_country_region.pivot(index="Country", columns="Region", values="Correlation")

# Create a custom colorscale manually
colorscale = [
    [0, "#0000FF"],  # Dark Blue
    [0.25, "#8080FF"],  # Light Blue
    [0.45, "#CCCCFF"],  # Very Light Blue
    [0.5, "#FFFFFF"],  # White
    [0.55, "#FFCCCC"],  # Very Light Red
    [0.75, "#FF8080"],  # Light Red
    [1, "#FF0000"]  # Dark Red
]

# Create the heatmap using Plotly
fig6 = go.Figure(data=go.Heatmap(
    z=pivot_corr_country_region.values,
    x=pivot_corr_country_region.columns,
    y=pivot_corr_country_region.index,
    colorscale=colorscale,
    zmin=-1,
    zmax=1,
    zmid=0,
    colorbar=dict(
        title='Correlation Coefficient',
        tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ticktext=['-1', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1']
    ),
    hovertemplate='Country: %{y}<br>Region: %{x}<br>Correlation: %{z:.2f}<extra></extra>'
))

# Update layout
fig6.update_layout(
    title='Correlation Between Extreme Weather Events and Economic Impact by Country and Region',
    xaxis_title='Region',
    yaxis_title='Country',
    width=1200,
    height=1000,
    xaxis=dict(tickangle=45),
    yaxis=dict(tickangle=0)
)

# Add annotations to display correlation coefficients
for i, row in enumerate(pivot_corr_country_region.values):
    for j, value in enumerate(row):
        if not np.isnan(value):  # Check if the value is not NaN
            fig6.add_annotation(
                x=pivot_corr_country_region.columns[j],
                y=pivot_corr_country_region.index[i],
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color='black' if abs(value) < 0.5 else 'white', size=8)
            )

# Show the plot
st.plotly_chart(fig6, use_container_width=True)

# Interactive Plot 3: Economic Impact by Region  
# Group the data by Region and Crop_Type, summing the Economic_Impact
grouped_data = filtered_data.groupby(['Region', 'Crop_Type'])['Economic_Impact_Million_USD'].sum().reset_index()

# Calculate the total economic impact for each region
region_totals = grouped_data.groupby('Region')['Economic_Impact_Million_USD'].sum().reset_index(name='Total_Impact')

# Merge the totals back to the grouped data
grouped_data = pd.merge(grouped_data, region_totals, on='Region')

# Calculate the percentage for each crop type within its region
grouped_data['Percentage'] = (grouped_data['Economic_Impact_Million_USD'] / grouped_data['Total_Impact']) * 100

# Create the initial stacked bar chart
fig3 = px.bar(grouped_data, x='Region', y='Economic_Impact_Million_USD', color='Crop_Type',
              title=f"{selected_country} Economic Impact by Region and Crop Type")

# Customize the layout
fig3.update_layout(
    width=800,  # Adjust the width of the chart
    height=500,  # Adjust the height of the chart
    barmode='stack',
    yaxis_title='Economic Impact (Million USD)',
    legend_title='Crop Type',
    hovermode='x unified'
)

# Add percentage annotations
annotations = []
for r in grouped_data.Region.unique():
    region_data = grouped_data[grouped_data.Region == r]
    cum_sum = 0
    for i, row in region_data.iterrows():
        annotations.append(
            dict(
                x=r,
                y=cum_sum + (row['Economic_Impact_Million_USD'] / 2),
                text=f"{row['Percentage']:.1f}%",
                showarrow=False,
                font=dict(color='white', size=10)
            )
        )
        cum_sum += row['Economic_Impact_Million_USD']

fig3.update_layout(annotations=annotations)

# Display the chart
st.subheader(f"{selected_country} Economic Impact")
st.plotly_chart(fig3, use_container_width=True)

# Interactive Plot 4: Extreme Weather Events vs Crop Yield
st.subheader(f"Extreme Weather Events vs Crop Yield ({selected_country})")
fig4 = px.scatter(filtered_data, x='Extreme_Weather_Events', y='Crop_Yield_MT_per_HA', 
                  size='Total_Precipitation_mm', color='Crop_Type', hover_name='Region',
                  title="Extreme Weather Events vs Crop Yield")
fig4.update_layout(yaxis_title='Crop Yield (MT Per HA)')
st.plotly_chart(fig4)



# Interactive Plot 5: Average CO2 Emissions Over Time by Country
st.subheader("Animated Time Series: Average CO2 Emissions Over Time by Country")
# Calculate average CO2 emissions per country per year
climate_impact_df_avg = data.groupby(['Country', 'Year'])['CO2_Emissions_MT'].mean().reset_index()

# Sort the DataFrame by Year to ensure sequential display
climate_impact_df_avg = climate_impact_df_avg.sort_values('Year')

# Calculate the overall min and max values for CO2 emissions
min_emissions = climate_impact_df_avg['CO2_Emissions_MT'].min()
max_emissions = climate_impact_df_avg['CO2_Emissions_MT'].max()

# Create a choropleth plot with time series and fixed color scale range
fig5 = px.choropleth(
    climate_impact_df_avg,
    locations='Country', 
    locationmode='country names',
    color='CO2_Emissions_MT',
    hover_name='Country',
    animation_frame='Year',
    color_continuous_scale='Viridis_r',
    range_color=[min_emissions, max_emissions],  # Set fixed range for color scale
    projection='natural earth',
    title='Average CO2 Emissions Over Time by Country'
)

# Update layout for better appearance
fig5.update_layout(
    width=1200,
    height=800,
    coloraxis_colorbar=dict(
        title="Average CO2 Emissions (MT)",
    ),
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Add slider
fig5.update_layout(
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Year:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {
                'args': [[f'{year}'], {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }],
                'label': f'{year}',
                'method': 'animate'
            } for year in climate_impact_df_avg['Year'].unique()
        ]
    }]
)

# Show the plot
st.plotly_chart(fig5)


# Interactive Plot 7: Average CO2 Emissions by Country Over Time
# Calculate average CO2 emissions per country per year
st.subheader('Average CO2 Emissions by Country Over Time')
climate_impact_df_avg = data.groupby(['Country', 'Year'])['CO2_Emissions_MT'].mean().reset_index()

# Sort the DataFrame by Year to ensure sequential display
climate_impact_df_avg = climate_impact_df_avg.sort_values('Year')

# Create a line chart with CO2 emissions over time
fig7 = px.line(
    climate_impact_df_avg,
    x='Year', 
    y='CO2_Emissions_MT',
    color='Country',  # Line color by country
    title=(f'Average CO2 Emissions by {selected_country} Over Time'),
    labels={'CO2_Emissions_MT': 'CO2 Emissions (Mt)', 'Year': 'Year'}
)

# Add a dropdown menu to filter by country
fig7.update_layout(
    updatemenus=[
        {
            'buttons': [
                {
                    'args': [{'visible': [country == selected_country for country in climate_impact_df_avg['Country'].unique()]}],
                    'label': selected_country,
                    'method': 'update'
                }
                for selected_country in climate_impact_df_avg['Country'].unique()
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }
    ]
)
st.plotly_chart(fig7)

# Interactive Plot 8: Average CO2 Emissions by Country Over Time
st.subheader('Average CO2 Emissions by Country Over Time (OWID Data)')
# Sort the DataFrame by 'year' to ensure proper animation order
cleaned_df = cleaned_df.sort_values('year')

# Create the choropleth map with animation
fig8 = px.choropleth(
    cleaned_df,
    locations='country',
    locationmode='country names',
    color='co2',
    hover_name='country',
    animation_frame='year',
    color_continuous_scale='Viridis_r',
    projection='natural earth',
    title='CO2 Emissions by Country Over Time (OWID Data)',
)

# Update layout for better appearance and adjust animation speed
fig8.update_layout(
    width=1000,
    height=600,
    coloraxis_colorbar=dict(
        title="CO2 Emissions",
        ticksuffix=" Mt",  # Correct unit based on your context
        tickformat=",.0f"
    ),
    updatemenus=[
        {
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100},  # Adjust duration here (in milliseconds)
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'showactive': False,
            'x': 0.1,
            'y': 1.1
        }
    ]
)

# Show the figure
st.plotly_chart(fig8)

# Interactive Plot 9: Average CO2 Emissions by Country Over Time
st.subheader('Average CO2 Emissions by Country Over Time')
# Create a line chart with CO2 emissions over time
fig9 = px.line(
    cleaned_df,
    x='year', 
    y='co2',
    color='country',  # Line color by country
    title='Average CO2 Emissions by Country Over Time',
    labels={'co2': 'CO2 Emissions (MT)', 'year': 'Year'}
)
st.plotly_chart(fig9)
# Displaying filter details
st.sidebar.write("Filters applied: Country - {}, Region - {}, Crop Type - {}, Year Range - {}".format(selected_country, selected_region, selected_crop_type, year_range))



