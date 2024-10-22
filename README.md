# Team6 Project 3: Analyzing the Impact of Climate Change on Agriculture

## Project Overview

This project leverages a Kaggle dataset (and a supplemental Our World in Data set) to investigate the impact of climate change on agriculture. Specifically, we analyze trends in temperature and crop yield, explore the relationship between extreme weather events and economic outcomes, and identify which crops have the largest economic impact on different countries. We also look at key indicators of climate change including global CO2 and Greenhouse Gas emissions over time (predicted, actual, and goals), and global average temperatues over time.

## Steps for Analysis

### 1. Aggregate and Analyze Trends

We begin by aggregating the mean values of continuous variables over time, with a focus on average temperature, crop yield, CO2 and GHG emissions.

### 2. Data Visualization

Data visualization is key to understanding trends over time. The visualizations include:

* **Overall trends** : Line plots showing the raw data for average temperature and crop yield over the entire time period.
* **5-Year Moving Average** : Visualizations using moving averages to smooth short-term fluctuations and reveal long-term trends. This technique is particularly useful when dealing with data that has large year-to-year variations.
* **Overlay of Raw Data and Moving Average** : Combined plots to allow for direct comparison between raw data and moving averages.
* **Geo-plots and animations**: Visualization to show global emissions.

### 3. Subplots

We utilize **plotly** to create subplots, grouping multiple visualizations within a single figure for easier comparison and analysis.

## Correlation Analysis

### Pearson Correlation

* **Pearson Correlation Coefficient** : `0.246`
* **Interpretation** : This suggests a weak positive linear relationship between average temperature and crop yield. As temperature increases, crop yield increases slightly, but the relationship is not strong.
* **P-value** : `0.182` indicates that the correlation is not statistically significant (p > 0.05).

### Spearman Correlation

* **Spearman Correlation Coefficient** : `0.175`
* **Interpretation** : A weak positive monotonic relationship between temperature and crop yield.
* **P-value** : `0.348`, indicating that the relationship is not statistically significant.

### Conclusion:

Both correlation coefficients suggest weak relationships, and the high p-values indicate that any observed correlations may not be statistically significant.

### Next Steps:

1. **Increase sample size** to detect stronger correlations.
2. **Explore non-linear models** like polynomial regression or decision trees.
3. **Consider other variables** such as rainfall, soil quality, and fertilizer use.

## Economic Impact Analysis

### Economic Impact of Crop Type by Country

We use a **heatmap** to visualize the economic impact of different crop types across countries, helping to identify which crops contribute the most to each country's economy.

* **Global Findings** :
* Sugarcane has the highest economic impact worldwide.
* Nigeria benefits the most from an economic perspective at the country level.
* **Country-Specific Findings** :
* **Australia** derives the most benefit from fruit.
* **India** gains the highest economic impact from sugarcane.
* **Nigeria** experiences the greatest economic gains from corn.

### Investigating Extreme Weather Events and Economic Impact

We analyze the relationship between extreme weather events and economic outcomes, focusing on regional variations.

* **Australia (New South Wales)** : A strong negative correlation between extreme weather events and economic impact, suggesting that more extreme weather reduces economic gains.
* **China (Eastern Region)** : A strong positive correlation, where more extreme weather increases economic impact, an unexpected finding that suggests resilience or economic adaptation.

## OLS Regression Results

We conducted an OLS regression to identify significant predictors of economic outcomes:

* **Significant variables** : Average Temperature, Total Precipitation (mm), and CO2 Emissions (MT), with p-values < 0.05.
* **Non-significant variable** : Soil Health Index, with a p-value of 0.847, indicating that it is not statistically significant.

## Conclusion

This project reveals insights into how climate change affects agriculture, highlighting key trends and areas for further research, including the need to explore non-linear relationships and include additional environmental variables.

## References

1. World Air Quality Index Project. "Climate Change Impact on Agriculture." Kaggle, 2020. Accessed throughout project work. [https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture](https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture).
2. Hannah Ritchie and Max Roser. "CO2 Emissions."  *Our World in Data* . Accessed throughout project work. [https://ourworldindata.org/co2-emissions](https://ourworldindata.org/co2-emissions).

---

### General References

* Anthropic,  *Claude AI* . Accessed throughout project work. [https://www.anthropic.com](https://www.anthropic.com).
* OpenAI,  *ChatGPT* . Accessed throughout project work. [https://chat.openai.com](https://chat.openai.com).
* *Stack Overflow* . Accessed throughout project work. [https://stackoverflow.com](https://stackoverflow.com).

---

## Usage

`dashboard.py` is the main app for this project. To run the app on your local machine, type the following in your terminal (assuming you're in the same directory as `dashboard.py`):

`streamlit run dashboard.py`

This will launch the app in your default browser. Make sure you have [Streamlit]() installed first by running `pip install streamlit`.

The databases for this project are hosted in PostgreSQL, using an Amazon RDS cloud server for reliable and scalable storage. This allows the app to connect to a live database, enabling real-time data queries. The project was deployed on a [Google Cloud Platform (GCP)](https://cloud.google.com/) server, however that link is now deprecated.
