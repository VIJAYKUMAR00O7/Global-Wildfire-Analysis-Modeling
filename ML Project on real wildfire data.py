import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Step 1: Fetch Wildfire Data from NASA FIRMS
# -------------------------------------------------------
# NASA FIRMS provides near real-time fire data. We'll use the MODIS global fires from the last 24 hours.
firms_url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv"

print("Downloading NASA FIRMS data...")
r = requests.get(firms_url)
if r.status_code == 200:
    with open("MODIS_C6_1_Global_24h.csv", 'wb') as f:
        f.write(r.content)
    print("NASA FIRMS data downloaded.")
else:
    raise Exception("Failed to download FIRMS data.")

fires_df = pd.read_csv("MODIS_C6_1_Global_24h.csv")
print("Fire Data Sample:")
print(fires_df.head())

# -------------------------------------------------------
# Step 2: Fetch Weather Data from Open-Meteo
# -------------------------------------------------------
# We'll fetch current weather data for a sample of fire coordinates.

fires_sample = fires_df.head(30).copy()  # Taking a smaller sample for demonstration
weather_data = []

print("\nFetching weather data from Open-Meteo...")
for idx, row in fires_sample.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    
    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,winddirection_10m,windspeed_10m"
    )
    w_resp = requests.get(weather_url)
    if w_resp.status_code == 200:
        w_data = w_resp.json()
        if 'hourly' in w_data and 'temperature_2m' in w_data['hourly']:
            # Take the mean of the hourly data as a simple summary
            temperature = pd.Series(w_data['hourly']['temperature_2m']).mean()
            humidity = pd.Series(w_data['hourly']['relativehumidity_2m']).mean()
            windspeed = pd.Series(w_data['hourly']['windspeed_10m']).mean()
            winddir = pd.Series(w_data['hourly']['winddirection_10m']).mean()
            
            weather_data.append({
                'latitude': lat,
                'longitude': lon,
                'temperature_2m_mean': temperature,
                'relativehumidity_2m_mean': humidity,
                'windspeed_10m_mean': windspeed,
                'winddirection_10m_mean': winddir
            })
        else:
            # If no hourly data is returned for some reason, append Nones
            weather_data.append({
                'latitude': lat,
                'longitude': lon,
                'temperature_2m_mean': None,
                'relativehumidity_2m_mean': None,
                'windspeed_10m_mean': None,
                'winddirection_10m_mean': None
            })
    else:
        weather_data.append({
            'latitude': lat,
            'longitude': lon,
            'temperature_2m_mean': None,
            'relativehumidity_2m_mean': None,
            'windspeed_10m_mean': None,
            'winddirection_10m_mean': None
        })

weather_df = pd.DataFrame(weather_data)
print("Weather Data Sample:")
print(weather_df.head())

# -------------------------------------------------------
# Step 3: Merge Fire Data and Weather Data
# -------------------------------------------------------
# Because we're dealing with floating point coordinates, let's round them to simplify the merge.
# In a real scenario, you'd want a more robust spatial join.
fires_sample['lat_rounded'] = fires_sample['latitude'].round(2)
fires_sample['lon_rounded'] = fires_sample['longitude'].round(2)
weather_df['lat_rounded'] = weather_df['latitude'].round(2)
weather_df['lon_rounded'] = weather_df['longitude'].round(2)

merged_df = pd.merge(
    fires_sample,
    weather_df.drop(columns=['latitude','longitude']),
    on=['lat_rounded', 'lon_rounded'],
    how='inner'
)

# Create a binary target: brightness > 330 = high intensity fire
merged_df['high_intensity'] = (merged_df['brightness'] > 330).astype(int)

# Select features for modeling
features = [
    'temperature_2m_mean',
    'relativehumidity_2m_mean',
    'windspeed_10m_mean',
    'winddirection_10m_mean',
    'scan', # fire area measurement
    'track' # fire area measurement
]
merged_df = merged_df.dropna(subset=features + ['high_intensity'])

X = merged_df[features]
y = merged_df['high_intensity']

print("\nMerged Data Sample:")
print(merged_df.head())

# -------------------------------------------------------
# Step 4: Train a Simple Model
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# Step 5: Visualization
# -------------------------------------------------------
# Plot feature importances
importances = model.feature_importances_
plt.figure(figsize=(8,4))
plt.bar(features, importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Demonstration complete.")

import seaborn as sns
import numpy as np

# ---------------------------------------
# 1. Feature Correlation Analysis
# ---------------------------------------
# Include the target in the correlation analysis
analysis_df = merged_df[features + ['high_intensity']].copy()
corr = analysis_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix with Target")
plt.show()

# Interpretation:
# This shows how each feature correlates with the target (high_intensity).
# Strong positive or negative correlations provide insights into which features are most related to fire intensity.

# ---------------------------------------
# 2. Confusion Matrix Visualization
# ---------------------------------------
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Intensity','High Intensity'], yticklabels=['Low Intensity','High Intensity'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Interpretation:
# This helps understand the type of errors the model makes.
# True Negatives (top-left), False Positives (top-right), False Negatives (bottom-left), and True Positives (bottom-right).

# ---------------------------------------
# 3. Probability Distributions & Histograms
# ---------------------------------------
# Let's visualize the distribution of brightness for fires classified as high vs low intensity in the dataset.
plt.figure(figsize=(8,5))
sns.histplot(data=merged_df, x='brightness', hue='high_intensity', kde=True, palette='viridis', multiple='stack')
plt.title("Distribution of Brightness by Fire Intensity Class")
plt.xlabel("Brightness")
plt.ylabel("Count")
plt.show()

# Interpretation:
# This shows how brightness separates the two classes and may indicate how well brightness alone could serve as a predictive feature.

# ---------------------------------------
# Optional: Model Explainability with SHAP
# ---------------------------------------
# SHAP provides detailed explanations for each feature’s contribution.
# Install SHAP if needed: pip install shap
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot of SHAP values for feature importance
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values[1], X_test)  # Plot for the 'high intensity' class
plt.show()

# Interpretation:
# The SHAP summary plot shows which features push the prediction towards high intensity (positive SHAP values) or away from it (negative SHAP values).

# ---------------------------------------
# Optional: Geospatial Visualization
# ---------------------------------------
# If you want to visualize the data points on a map. This requires geopandas and a basemap or contextily for background tiles.
# pip install geopandas contextily
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# Create a GeoDataFrame with fire points
geometry = [Point(xy) for xy in zip(merged_df['longitude'], merged_df['latitude'])]
gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

# Convert to a Web Mercator projection for plotting with contextily
gdf = gdf.to_crs(epsg=3857)

plt.figure(figsize=(10,8))
ax = gdf.plot(column='high_intensity', cmap='coolwarm', markersize=5, legend=True)
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)  # Adds a basemap
plt.title("Geographical Distribution of Fires by Intensity")
plt.show()

# Interpretation:
# Points on the map show where fires occur. Colors indicate intensity class. 
# This helps understand spatial patterns or cluster locations.

import geopandas as gpd
import contextily as ctx
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import numpy as np

# ---------------------------------------
# Assumes `merged_df` is available with 
# columns: 'latitude','longitude','brightness'
# ---------------------------------------

# Convert merged_df to a GeoDataFrame
geometry = [Point(xy) for xy in zip(merged_df['longitude'], merged_df['latitude'])]
gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

# Load world boundaries shapefile
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# Spatially join fire points to countries
fires_with_country = gpd.sjoin(gdf, world, how="left", predicate="intersects")

# Count fires per country
fires_per_country = fires_with_country.groupby('ADMIN').size().reset_index(name='count')
fires_per_country = fires_per_country.sort_values('count', ascending=False)

# Compute average brightness per country
avg_brightness_per_country = fires_with_country.groupby('ADMIN')['brightness'].mean().reset_index(name='avg_brightness')
avg_brightness_per_country = avg_brightness_per_country.sort_values('avg_brightness', ascending=False)

# ---------------------------------------
# Visualization 1: Global distribution of fires
# ---------------------------------------
fires_web_mercator = fires_with_country.to_crs(epsg=3857)
world_web_mercator = world.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(15,10))
world_web_mercator.plot(ax=ax, color='white', edgecolor='black')
fires_web_mercator.plot(ax=ax, markersize=5, color='red', alpha=0.5)
ax.set_title("Global Distribution of Detected Fires")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=1)
plt.show()

# ---------------------------------------
# Visualization 2: Top Countries by Fire Count (Let's show Top 6 here)
# ---------------------------------------
top_6_count = fires_per_country.head(6)

plt.figure(figsize=(10,6))
sns.barplot(data=top_6_count, x='count', y='ADMIN', palette='Reds_r')
plt.title("Top 6 Countries by Number of Fires")
plt.xlabel("Number of Fires")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Visualization 3: Average Brightness by Country (Choropleth)
# ---------------------------------------
world_with_fires = world.merge(avg_brightness_per_country, left_on='ADMIN', right_on='ADMIN', how='left')
world_with_fires_plot = world_with_fires.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(15,10))
world_with_fires_plot.plot(column='avg_brightness', ax=ax, legend=True, cmap='YlOrRd', missing_kwds={"color": "lightgrey"})
ax.set_title("Average Fire Brightness by Country")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=1)
plt.show()

# ---------------------------------------
# Print summaries for top 6 countries
# ---------------------------------------
print("Top 6 Countries by Fire Count:")
print(top_6_count)

top_6_brightness = avg_brightness_per_country.head(6)
print("\nTop 6 Countries by Average Fire Brightness:")
print(top_6_brightness)


# ---------------------------------------
# Country-Level Aggregation and Insights: Examining Wildfire Patterns Across the Globe
# ---------------------------------------

# This section assumes that you have already run the initial steps to produce:
#  - merged_df (with columns 'latitude', 'longitude', 'brightness')
#  - A world boundaries shapefile (e.g. ne_110m_admin_0_countries.shp) in the current directory
#
# The code below:
# - Joins each fire detection to a country.
# - Computes the top 6 countries by the number of fires detected.
# - Computes the top 6 countries by average brightness of fires.
# - Prints these top countries, highlighting India if it is included.
# - Provides descriptive commentary to help interpret the results.

import geopandas as gpd
import contextily as ctx
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import numpy as np

# Convert merged_df to a GeoDataFrame using fire coordinates
geometry = [Point(xy) for xy in zip(merged_df['longitude'], merged_df['latitude'])]
gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

# Load world boundaries shapefile
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# Spatially join fire points to countries
fires_with_country = gpd.sjoin(gdf, world, how="left", predicate="intersects")

# Count fires per country
fires_per_country = fires_with_country.groupby('ADMIN').size().reset_index(name='count')
fires_per_country = fires_per_country.sort_values('count', ascending=False)

# Compute average brightness per country
avg_brightness_per_country = fires_with_country.groupby('ADMIN')['brightness'].mean().reset_index(name='avg_brightness')
avg_brightness_per_country = avg_brightness_per_country.sort_values('avg_brightness', ascending=False)

# Display the top 6 countries by fire count and average brightness
top_6_count = fires_per_country.head(6)
top_6_brightness = avg_brightness_per_country.head(6)

# ---------------------------------------
# Visualization 1: Global distribution of fires
# ---------------------------------------
# Plot a world map showing all detected fires. This gives a global snapshot of where fires are occurring.
fires_web_mercator = fires_with_country.to_crs(epsg=3857)
world_web_mercator = world.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(15,10))
world_web_mercator.plot(ax=ax, color='white', edgecolor='black')
fires_web_mercator.plot(ax=ax, markersize=5, color='red', alpha=0.5)
ax.set_title("Global Distribution of Detected Fires")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=1)
plt.show()

# ---------------------------------------
# Visualization 2: Top 6 Countries by Fire Count
# ---------------------------------------
# A bar chart of the top 6 countries with the most detected fires.
plt.figure(figsize=(10,6))
sns.barplot(data=top_6_count, x='count', y='ADMIN', palette='Reds_r')
plt.title("Top 6 Countries by Number of Fires")
plt.xlabel("Number of Fires")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Visualization 3: Average Brightness by Country (Choropleth)
# ---------------------------------------
# A choropleth map illustrating the average brightness of fires by country.
# Higher brightness can indicate more intense fires, so this map helps identify regions where fires may be particularly severe.
world_with_fires = world.merge(avg_brightness_per_country, left_on='ADMIN', right_on='ADMIN', how='left')
world_with_fires_plot = world_with_fires.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(15,10))
world_with_fires_plot.plot(column='avg_brightness', ax=ax, legend=True, cmap='YlOrRd', missing_kwds={"color": "lightgrey"})
ax.set_title("Average Fire Brightness by Country")
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=1)
plt.show()

# ---------------------------------------
# Print out summaries with descriptive commentary
# ---------------------------------------
print("The following lists highlight which countries lead in total fire counts and which see the brightest (potentially most intense) fires.")

print("\nTop 6 Countries by Fire Count:")
count_countries = top_6_count['ADMIN'].values.tolist()
for i, country in enumerate(count_countries, start=1):
    # Highlight if it's India
    if country == "India":
        print(f"{i}. {country} (INDIA)")
    else:
        print(f"{i}. {country}")
print("Countries at the top of this list may have widespread environmental conditions favorable to fire occurrences, such as dry climates, extensive vegetation, or human activities contributing to fires.")

print("\nTop 6 Countries by Average Fire Brightness:")
brightness_countries = top_6_brightness['ADMIN'].values.tolist()
for i, country in enumerate(brightness_countries, start=1):
    # Highlight if it's India
    if country == "India":
        print(f"{i}. {country} (INDIA)")
    else:
        print(f"{i}. {country}")
print("Countries with higher average brightness values may experience more intense fires. This could reflect hotter burning conditions, dense fuel loads, or other factors making fires more severe.")

print("\nThese insights help identify where fires are occurring most frequently and where they might be most intense. Policymakers, environmental agencies, and fire management teams can use this information to allocate resources, plan firefighting strategies, and prioritize fire prevention measures in the most affected regions.")

