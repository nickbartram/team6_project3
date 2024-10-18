# Team6 Project 3: Analyzing the Impact of Climate Change on Agriculture

## Project Overview

This project leverages a Kaggle dataset to investigate the impact of climate change on agriculture. Specifically, we analyze trends in temperature and crop yield, explore the relationship between extreme weather events and economic outcomes, and identify which crops have the largest economic impact on different countries.

## Steps for Analysis

### 1. Aggregate and Analyze Trends

We begin by aggregating the mean values of continuous variables over time, with a focus on average temperature and crop yield.

### 2. Data Visualization

Data visualization is key to understanding trends over time. The visualizations include:

* **Overall trends** : Line plots showing the raw data for average temperature and crop yield over the entire time period.
* **5-Year Moving Average** : Visualizations using moving averages to smooth short-term fluctuations and reveal long-term trends. This technique is particularly useful when dealing with data that has large year-to-year variations.
* **Overlay of Raw Data and Moving Average** : Combined plots to allow for direct comparison between raw data and moving averages.

### 3. Subplots

We utilize **matplotlib** to create subplots, grouping multiple visualizations within a single figure for easier comparison and analysis.

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
