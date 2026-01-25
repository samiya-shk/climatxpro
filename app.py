from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import requests
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')

# Make sure templates folder exists
if not os.path.exists('templates'):
    os.makedirs('templates')

# Path to your data file
COMBINED_CSV = os.path.join(os.path.dirname(__file__), "data", "combined_data_with_events.csv")
DATA_PATH = COMBINED_CSV  # Use the same CSV for prediction

# WeatherAPI key
WEATHER_API_KEY = '499f98d722184220a18161943252808'  # User's API key
# Mumbai coordinates
LAT = 19.0760
LON = 72.8777

# ---------------------------
# Data loading + cleaning
# ---------------------------
def load_and_prepare():
    """
    Loads and prepares the combined_data_with_events.csv file.
    """
    try:
        # Try different encodings to handle special characters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(COMBINED_CSV, quotechar='"', quoting=csv.QUOTE_ALL, encoding=encoding)
                print(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("Could not read CSV with any of the provided encodings")
            
    except FileNotFoundError:
        raise FileNotFoundError("⚠️ combined_data_with_events.csv not found in project folder")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    # Basic cleanup
    if "Year" not in df.columns:
        raise ValueError("⚠️ 'Year' column missing in combined_data_with_events.csv")

    # Drop rows with missing Year
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # Convert numeric columns safely
    for col in df.columns:
        if col not in ["Year", "Event_Notes", "Impact_Observed"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix temperature column names if they have encoding issues
    temp_max_col = "Avg Temp Max (Â°C)"
    temp_min_col = "Avg Temp Min (Â°C)"
    
    # Try alternative column names if the standard ones don't exist
    if temp_max_col not in df.columns:
        # Look for similar column names
        for col in df.columns:
            if "temp" in col.lower() and "max" in col.lower():
                temp_max_col = col
                break
    
    if temp_min_col not in df.columns:
        # Look for similar column names
        for col in df.columns:
            if "temp" in col.lower() and "min" in col.lower():
                temp_min_col = col
                break
    
    # Check if we found both temperature columns
    if temp_max_col in df.columns and temp_min_col in df.columns:
        # Calculate average temperature
        df["Temperature"] = (df[temp_max_col] + df[temp_min_col]) / 2
    else:
        # If we can't find temperature columns, create an empty one
        df["Temperature"] = np.nan
        print("Warning: Could not find temperature columns")
    
    # Create a cleaner dataframe with the columns we need
    columns_needed = [
        "Year", 
        "Temperature", 
        "Total Rainfall (mm)", 
        "Relative Humidity (%)", 
        "CO2 Emissions Mumbai (kt)", 
        "Annual Windspeed",
        "Event_Notes",
        "Impact_Observed"
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in columns_needed if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Rename columns for easier access
    column_mapping = {
        "Total Rainfall (mm)": "Rainfall",
        "Relative Humidity (%)": "Humidity",
        "CO2 Emissions Mumbai (kt)": "CO2",
        "Annual Windspeed": "WindSpeed"
    }
    
    # Only rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean = df_clean.rename(columns={old_name: new_name})
    
    return df_clean

# ---------------------------
# Get weather forecast from WeatherAPI
# ---------------------------
def get_weather_forecast():
    """
    Get 7-day weather forecast from WeatherAPI
    """
    try:
        # Get current weather
        location = f"{LAT},{LON}"
        current_url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=yes"
        
        current_response = requests.get(current_url)
        if current_response.status_code != 200:
            return None, f"Current weather API request failed with status code {current_response.status_code}"
        
        current_data = current_response.json()
        
        # Get forecast
        forecast_url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={location}&days=7&aqi=no&alerts=no"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code != 200:
            return None, f"Forecast API request failed with status code {forecast_response.status_code}"
        
        forecast_data = forecast_response.json()
        
        # Process current weather
        current_weather = {
            'temp_c': current_data['current']['temp_c'],
            'condition': current_data['current']['condition']['text'],
            'icon': current_data['current']['condition']['icon'],
            'humidity': current_data['current']['humidity'],
            'wind_kph': current_data['current']['wind_kph'],
            'pressure_mb': current_data['current']['pressure_mb'],
            'feelslike_c': current_data['current']['feelslike_c'],
            'uv': current_data['current'].get('uv', 0),
            'last_updated': current_data['current']['last_updated']
        }
        
        # Process daily forecast data
        daily_forecasts = []
        for day in forecast_data['forecast']['forecastday']:
            date = day['date']
            day_data = day['day']
            
            forecast = {
                'date': date,
                'day_name': datetime.strptime(date, '%Y-%m-%d').strftime('%A'),
                'max_temp_c': day_data['maxtemp_c'],
                'min_temp_c': day_data['mintemp_c'],
                'avg_temp_c': day_data['avgtemp_c'],
                'max_wind_kph': day_data['maxwind_kph'],
                'total_precip_mm': day_data['totalprecip_mm'],
                'avg_humidity': day_data['avghumidity'],
                'daily_will_it_rain': day_data['daily_will_it_rain'],
                'daily_chance_of_rain': day_data['daily_chance_of_rain'],
                'condition': day_data['condition']['text'],
                'icon': day_data['condition']['icon'],
                'uv_index': day_data['uv']
            }
            
            # Add hourly data for more detailed visualization
            hourly_data = []
            for hour in day['hour']:
                hourly_data.append({
                    'time': hour['time'],
                    'temp_c': hour['temp_c'],
                    'will_it_rain': hour['will_it_rain'],
                    'chance_of_rain': hour['chance_of_rain'],
                    'humidity': hour['humidity']
                })
            
            forecast['hourly'] = hourly_data
            daily_forecasts.append(forecast)
        
        return {
            'current': current_weather,
            'forecast': daily_forecasts,
            'location': forecast_data['location']
        }, None
    
    except Exception as e:
        return None, f"Error fetching weather data: {str(e)}"

# ---------------------------
# Compute descriptive stats
# ---------------------------
def descriptive_stats(df):
    metrics = ["Rainfall", "Temperature", "Humidity", "CO2", "WindSpeed"]
    stats_rows = []
    for m in metrics:
        if m in df.columns and df[m].dropna().size > 0:
            s = df[m].dropna()
            stats_rows.append({
                "Parameter": m,
                "Mean": round(s.mean(), 3),
                "Median": round(s.median(), 3),
                "StdDev": round(s.std(), 3),
                "Min": round(s.min(), 3),
                "Max": round(s.max(), 3)
            })
    stats_df = pd.DataFrame(stats_rows)
    return stats_df

# ---------------------------
# Extreme events & anomalies
# ---------------------------
def detect_extremes(df):
    out = {}
    # safe checks
    if "Rainfall" in df.columns and not df["Rainfall"].dropna().empty:
        idx = df["Rainfall"].idxmax()
        out["MaxRainfallYear"] = int(df.loc[idx, "Year"])
        out["MaxRainfallValue"] = float(df.loc[idx, "Rainfall"])
        idx_min = df["Rainfall"].idxmin()
        out["MinRainfallYear"] = int(df.loc[idx_min, "Year"])
        out["MinRainfallValue"] = float(df.loc[idx_min, "Rainfall"])
    if "Temperature" in df.columns and not df["Temperature"].dropna().empty:
        idx = df["Temperature"].idxmax()
        out["MaxTempYear"] = int(df.loc[idx, "Year"])
        out["MaxTempValue"] = float(df.loc[idx, "Temperature"])
        idx_min = df["Temperature"].idxmin()
        out["MinTempYear"] = int(df.loc[idx_min, "Year"])
        out["MinTempValue"] = float(df.loc[idx_min, "Temperature"])
    if "CO2" in df.columns and not df["CO2"].dropna().empty:
        idx = df["CO2"].idxmax()
        out["MaxCO2Year"] = int(df.loc[idx, "Year"])
        out["MaxCO2Value"] = float(df.loc[idx, "CO2"])
        idx_min = df["CO2"].idxmin()
        out["MinCO2Year"] = int(df.loc[idx_min, "Year"])
        out["MinCO2Value"] = float(df.loc[idx_min, "CO2"])
    
    # Get flood years based on impact observations
    if "Impact_Observed" in df.columns:
        # Check if column contains string values
        if df["Impact_Observed"].dtype == object:
            flood_years = df[df["Impact_Observed"].str.contains("flood", case=False, na=False)]["Year"].tolist()
            out["FloodYears"] = flood_years
    
    return out

# ---------------------------
# Auto-narrative (What / Why / Consequence)
# ---------------------------
def narrative_insights(df, stats_df, extremes):
    narrative = {"What": "", "Why": "", "Consequences": ""}

    # WHAT: high-level changes
    if "Temperature" in df.columns and df["Temperature"].dropna().size > 1:
        change = df["Temperature"].dropna().iloc[-1] - df["Temperature"].dropna().iloc[0]
        narrative["What"] += f"Temperature change over dataset: {change:.2f}°C. "
    else:
        narrative["What"] += "Insufficient temperature data for trend analysis. "

    if "Rainfall" in df.columns and df["Rainfall"].dropna().size > 1:
        change_r = df["Rainfall"].dropna().iloc[-1] - df["Rainfall"].dropna().iloc[0]
        narrative["What"] += f"Rainfall change over dataset: {change_r:.1f} mm. "
    else:
        narrative["What"] += "Insufficient rainfall data for trend analysis. "

    # WHY: simple correlation-based interpretation
    numeric = df[["Temperature", "Rainfall", "Humidity", "CO2", "WindSpeed"]].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if "CO2" in numeric.columns and "Temperature" in numeric.columns:
        corr = numeric[["CO2","Temperature"]].corr().iloc[0,1]
        if pd.notna(corr):
            if corr > 0.5:
                narrative["Why"] = f"CO₂ and Temperature show a strong positive correlation ({corr:.2f}), suggesting greenhouse forcing influence."
            elif corr > 0.25:
                narrative["Why"] = f"CO₂ and Temperature show moderate correlation ({corr:.2f})."
            else:
                narrative["Why"] = f"CO₂ and Temperature correlation is weak ({corr:.2f})."
        else:
            narrative["Why"] = "Insufficient CO₂-temperature data for correlation."
    else:
        narrative["Why"] = "CO₂ or Temperature data not sufficient to compute correlation."

    # Additional why: rainfall vs humidity
    if "Rainfall" in numeric.columns and "Humidity" in numeric.columns:
        corr_rh = numeric[["Rainfall","Humidity"]].corr().iloc[0,1]
        if pd.notna(corr_rh) and corr_rh > 0.4:
            narrative["Why"] += f" Rainfall correlates with Humidity (r={corr_rh:.2f})."

    # CONSEQUENCES: map extremes to likely impacts
    consequences = []
    if extremes.get("MaxRainfallYear"):
        # Get event and impact for max rainfall year
        max_rain_year = extremes["MaxRainfallYear"]
        year_data = df[df["Year"] == max_rain_year]
        if not year_data.empty:
            event = year_data["Event_Notes"].iloc[0] if pd.notna(year_data["Event_Notes"].iloc[0]) else ""
            impact = year_data["Impact_Observed"].iloc[0] if pd.notna(year_data["Impact_Observed"].iloc[0]) else ""
            
            consequence = f"High flood risk observed in {max_rain_year} (peak rainfall {extremes['MaxRainfallValue']:.1f} mm)."
            if event:
                consequence += f" Event: {event}."
            if impact:
                consequence += f" Impact: {impact}."
            consequences.append(consequence)
    
    if extremes.get("MaxTempYear"):
        # Get event and impact for max temp year
        max_temp_year = extremes["MaxTempYear"]
        year_data = df[df["Year"] == max_temp_year]
        if not year_data.empty:
            event = year_data["Event_Notes"].iloc[0] if pd.notna(year_data["Event_Notes"].iloc[0]) else ""
            impact = year_data["Impact_Observed"].iloc[0] if pd.notna(year_data["Impact_Observed"].iloc[0]) else ""
            
            consequence = f"Extreme heat recorded in {max_temp_year} ({extremes['MaxTempValue']:.1f}°C) — public health concerns."
            if event:
                consequence += f" Event: {event}."
            if impact:
                consequence += f" Impact: {impact}."
            consequences.append(consequence)
    
    if narrative["Why"]:
        consequences.append("Rising CO₂ and temperature trends can intensify heatwaves and modify rainfall extremes.")
    
    # Add flood years if any
    if extremes.get("FloodYears"):
        flood_years = ", ".join(map(str, extremes["FloodYears"]))
        consequences.append(f"Years with recorded flood impacts: {flood_years}.")
    
    narrative["Consequences"] = " ".join(consequences) if consequences else "No immediate consequences found in dataset."

    return narrative

# ---------------------------
# Create trend + anomaly plot HTML for a metric
# ---------------------------
def plot_metric(df, col, title, zscore_threshold=2):
    # require Year and col
    if col not in df.columns or df[col].dropna().empty:
        return None
    
    # Create a clean dataframe with just the data we need
    plot_df = pd.DataFrame({
        "Year": df["Year"],
        "Value": df[col]
    }).dropna(subset=["Year"])
    
    plot_df = plot_df.sort_values("Year")
    
    # Define keywords for each metric type
    if col == "Rainfall":
        y_label = "Rainfall (mm)"
        metric_name = "Rainfall"
        unit = "mm"
        event_keywords = ["rain", "flood", "precipitation", "monsoon", "downpour", "rainfall"]
    elif col == "Humidity":
        y_label = "Humidity (%)"
        metric_name = "Humidity"
        unit = "%"
        event_keywords = ["humidity", "moisture", "drought", "dry", "humid"]
    elif col == "Temperature":
        y_label = "Temperature (°C)"
        metric_name = "Temperature"
        unit = "°C"
        event_keywords = ["temperature", "heat", "cold", "warm", "hot", "temp"]
    elif col == "CO2":
        y_label = "CO2 (kt)"
        metric_name = "CO2"
        unit = "kt"
        event_keywords = ["co2", "carbon", "emission"]
    elif col == "WindSpeed":
        y_label = "Wind Speed (km/h)"
        metric_name = "Wind Speed"
        unit = "km/h"
        event_keywords = ["wind", "storm", "cyclone", "breeze", "windy"]
    else:
        y_label = title
        metric_name = title
        unit = ""
        event_keywords = []
    
    # line plot
    fig = px.line(plot_df, x="Year", y="Value", title=title, markers=True)
    
    # Update y-axis label
    fig.update_yaxes(title=y_label)
    
    # Add event markers if any
    if "Event_Notes" in df.columns:
        event_df = df[df["Event_Notes"] != ""].copy()
        if not event_df.empty:
            # Check if events are relevant to this metric
            event_df["Relevant"] = event_df["Event_Notes"].apply(
                lambda x: any(keyword in str(x).lower() for keyword in event_keywords)
            )
            relevant_events = event_df[event_df["Relevant"]]
            
            if not relevant_events.empty:
                hover_texts = []
                for _, r in relevant_events.iterrows():
                    if unit:
                        value_text = f"{metric_name}: {r[col]:.1f} {unit}"
                    else:
                        value_text = f"{metric_name}: {r[col]}"
                    
                    ht = f"Year: {int(r['Year'])}<br>{value_text}"
                    ht += f"<br>Event: {r['Event_Notes']}"
                    
                    if "Impact_Observed" in df.columns and pd.notna(r["Impact_Observed"]) and r["Impact_Observed"] != "":
                        ht += f"<br>Impact: {r['Impact_Observed']}"
                    
                    hover_texts.append(ht)
                
                fig.add_trace(go.Scatter(
                    x=relevant_events["Year"],
                    y=relevant_events[col],
                    mode="markers",
                    marker=dict(color="orange", size=12, symbol="diamond"),
                    name="Event",
                    hovertext=hover_texts,
                    hoverinfo="text"
                ))
    
    # anomalies via zscore
    vals = plot_df["Value"].dropna()
    if len(vals) >= 3:
        z = stats.zscore(vals)
        # map back to indices in plot_df (only positions where non-null)
        nonnull_idx = plot_df["Value"].dropna().index
        anom_positions = nonnull_idx[np.where(np.abs(z) > zscore_threshold)[0]]
        if len(anom_positions) > 0:
            anom = plot_df.loc[anom_positions]
            hover_texts = []
            for _, r in anom.iterrows():
                # Create metric-specific hover text
                if unit:
                    value_text = f"{metric_name}: {r['Value']:.1f} {unit}"
                else:
                    value_text = f"{metric_name}: {r['Value']}"
                
                ht = f"Year: {int(r['Year'])}<br>{value_text}"
                
                # Get event and impact information for this year from the original data
                year_data = df[df["Year"] == r["Year"]]
                if not year_data.empty:
                    # Safely get Event_Notes and Impact_Observed
                    event_note = None
                    impact_observed = None
                    
                    if "Event_Notes" in year_data.columns:
                        event_note = year_data["Event_Notes"].iloc[0] if pd.notna(year_data["Event_Notes"].iloc[0]) and year_data["Event_Notes"].iloc[0] != '' and year_data["Event_Notes"].iloc[0] != 'nan' else None
                    
                    if "Impact_Observed" in year_data.columns:
                        impact_observed = year_data["Impact_Observed"].iloc[0] if pd.notna(year_data["Impact_Observed"].iloc[0]) and year_data["Impact_Observed"].iloc[0] != '' and year_data["Impact_Observed"].iloc[0] != 'nan' else None
                    
                    # Check if event note is relevant to this metric
                    if event_note:
                        event_note_str = str(event_note).lower()
                        if any(keyword in event_note_str for keyword in event_keywords):
                            ht += f"<br>Event: {event_note}"
                    
                    # Check if impact observed is relevant to this metric
                    if impact_observed:
                        impact_observed_str = str(impact_observed).lower()
                        if any(keyword in impact_observed_str for keyword in event_keywords):
                            ht += f"<br>Impact: {impact_observed}"
                
                hover_texts.append(ht)
            
            fig.add_trace(go.Scatter(
                x=anom["Year"],
                y=anom["Value"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Anomaly",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# ---------------------------
# Generate insights list
# ---------------------------
def generate_insights_list(df, extremes):
    insights = []
    
    # CO2 and temperature trends
    if "CO2" in df.columns and "Temperature" in df.columns:
        df_co2 = df[["Year", "CO2"]].dropna()
        df_temp = df[["Year", "Temperature"]].dropna()
        if len(df_co2) > 1 and len(df_temp) > 1:
            # Linear regression for CO2
            co2_slope, _, _, _, _ = stats.linregress(df_co2["Year"], df_co2["CO2"])
            temp_slope, _, _, _, _ = stats.linregress(df_temp["Year"], df_temp["Temperature"])
            if co2_slope > 0 and temp_slope > 0:
                insights.append(f"Both CO2 (slope: {co2_slope:.2f}) and temperature (slope: {temp_slope:.2f}°C/year) show increasing trends over the years.")
    
    # Extreme rainfall and humidity
    if "Rainfall" in df.columns and "Humidity" in df.columns:
        if extremes.get("MaxRainfallYear"):
            max_rain_year = extremes["MaxRainfallYear"]
            humidity_in_max_rain_year = df[df["Year"] == max_rain_year]["Humidity"].values
            if len(humidity_in_max_rain_year) > 0:
                humidity_in_max_rain_year = humidity_in_max_rain_year[0]
                avg_humidity = df["Humidity"].mean()
                if humidity_in_max_rain_year > avg_humidity:
                    insights.append(f"The year with maximum rainfall ({max_rain_year}) had humidity ({humidity_in_max_rain_year:.1f}%) above the average ({avg_humidity:.1f}%).")
    
    # CO2 extremes and temperature
    if extremes.get("MaxCO2Year") and "Temperature" in df.columns:
        max_co2_year = extremes["MaxCO2Year"]
        temp_in_max_co2_year = df[df["Year"] == max_co2_year]["Temperature"].values
        if len(temp_in_max_co2_year) > 0:
            temp_in_max_co2_year = temp_in_max_co2_year[0]
            avg_temp = df["Temperature"].mean()
            if temp_in_max_co2_year > avg_temp:
                insights.append(f"The year with maximum CO2 emissions ({max_co2_year}) had temperature ({temp_in_max_co2_year:.1f}°C) above the average ({avg_temp:.1f}°C).")
    
    # Flood years and rainfall
    if extremes.get("FloodYears"):
        flood_years = extremes["FloodYears"]
        if flood_years:
            # Get rainfall data for flood years
            flood_rainfalls = []
            for year in flood_years:
                rain_data = df[df["Year"] == year]["Rainfall"]
                if not rain_data.empty:
                    flood_rainfalls.append(rain_data.iloc[0])
            
            if flood_rainfalls:
                avg_flood_rainfall = sum(flood_rainfalls) / len(flood_rainfalls)
                avg_rainfall = df["Rainfall"].mean()
                if avg_flood_rainfall > avg_rainfall:
                    insights.append(f"Flood years had an average rainfall of {avg_flood_rainfall:.1f} mm, which is above the overall average of {avg_rainfall:.1f} mm.")
    
    # Temperature trend
    if "Temperature" in df.columns and df["Temperature"].dropna().size > 1:
        temp_trend = stats.linregress(df["Year"].dropna(), df["Temperature"].dropna())
        if temp_trend.slope > 0.01:
            insights.append(f"Temperature has been increasing at a rate of {temp_trend.slope:.3f}°C per year, indicating a clear warming trend.")
    
    # CO2 trend
    if "CO2" in df.columns and df["CO2"].dropna().size > 1:
        co2_trend = stats.linregress(df["Year"].dropna(), df["CO2"].dropna())
        if co2_trend.slope > 100:
            insights.append(f"CO2 emissions have been rising at approximately {co2_trend.slope:.0f} units per year, contributing to climate change impacts.")
    
    # Event frequency insights
    if "Event_Notes" in df.columns:
        event_count = df[df["Event_Notes"] != ""].shape[0]
        total_years = df.shape[0]
        if event_count > 0:
            insights.append(f"Significant climate events were recorded in {event_count} out of {total_years} years ({event_count/total_years*100:.1f}%).")
    
    return insights

# ---------------------------
# Root route - redirects to historical
# ---------------------------
@app.route("/")
def index():
    """Redirect to historical analysis page"""
    return redirect(url_for("historical"))

# ---------------------------
# Prediction Route
# ---------------------------
@app.route("/prediction")
def prediction():
    try:
        # Get weather forecast
        weather_data, error = get_weather_forecast()
        
        if error:
            return render_template("prediction.html", error=error)
        
        current = weather_data['current']
        forecast = weather_data['forecast']
        location = weather_data['location']
        
        # Create temperature graph
        temp_fig = go.Figure()
        
        # Add min and max temperature traces
        temp_fig.add_trace(go.Scatter(
            x=[day['date'] for day in forecast],
            y=[day['min_temp_c'] for day in forecast],
            mode='lines+markers',
            name='Min Temperature',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Min Temp: %{y}°C<extra></extra>'
        ))
        
        temp_fig.add_trace(go.Scatter(
            x=[day['date'] for day in forecast],
            y=[day['max_temp_c'] for day in forecast],
            mode='lines+markers',
            name='Max Temperature',
            line=dict(color='red'),
            hovertemplate='Date: %{x}<br>Max Temp: %{y}°C<extra></extra>'
        ))
        
        temp_fig.add_trace(go.Scatter(
            x=[day['date'] for day in forecast],
            y=[day['avg_temp_c'] for day in forecast],
            mode='lines+markers',
            name='Average Temperature',
            line=dict(color='orange'),
            hovertemplate='Date: %{x}<br>Avg Temp: %{y}°C<extra></extra>'
        ))
        
        temp_fig.update_layout(
            title="7-Day Temperature Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode='x unified'
        )
        
        # Create humidity graph
        hum_fig = go.Figure()
        
        hum_fig.add_trace(go.Scatter(
            x=[day['date'] for day in forecast],
            y=[day['avg_humidity'] for day in forecast],
            mode='lines+markers',
            name='Average Humidity',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>Humidity: %{y}%<extra></extra>'
        ))
        
        hum_fig.update_layout(
            title="7-Day Humidity Forecast",
            xaxis_title="Date",
            yaxis_title="Humidity (%)",
            hovermode='x unified'
        )
        
        # Create rain probability graph
        rain_fig = go.Figure()
        
        rain_fig.add_trace(go.Bar(
            x=[day['date'] for day in forecast],
            y=[day['daily_chance_of_rain'] for day in forecast],
            name='Rain Probability',
            marker_color='lightblue',
            hovertemplate='Date: %{x}<br>Rain Probability: %{y}%<extra></extra>'
        ))
        
        rain_fig.update_layout(
            title="7-Day Rain Probability",
            xaxis_title="Date",
            yaxis_title="Probability (%)",
            hovermode='x unified'
        )
        
        # Create hourly temperature graph for today
        hourly_temp_fig = go.Figure()
        today = forecast[0]
        hours = [h['time'] for h in today['hourly']]
        temps = [h['temp_c'] for h in today['hourly']]
        
        hourly_temp_fig.add_trace(go.Scatter(
            x=hours,
            y=temps,
            mode='lines+markers',
            name='Hourly Temperature',
            line=dict(color='purple'),
            hovertemplate='Time: %{x}<br>Temp: %{y}°C<extra></extra>'
        ))
        
        hourly_temp_fig.update_layout(
            title="Today's Hourly Temperature",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode='x unified'
        )
        
        # Convert figures to HTML
        temp_graph = temp_fig.to_html(full_html=False, include_plotlyjs='cdn')
        hum_graph = hum_fig.to_html(full_html=False, include_plotlyjs='cdn')
        rain_graph = rain_fig.to_html(full_html=False, include_plotlyjs='cdn')
        hourly_temp_graph = hourly_temp_fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Prepare additional insights
        insights = []
        
        # Find days with high rain probability
        rainy_days = [day for day in forecast if day['daily_chance_of_rain'] > 60]
        if rainy_days:
            rainy_day_names = [day['day_name'] for day in rainy_days]
            insights.append(f"High chance of rain expected on: {', '.join(rainy_day_names)}")
        
        # Find hottest and coolest days
        hottest_day = max(forecast, key=lambda x: x['max_temp_c'])
        coolest_day = min(forecast, key=lambda x: x['min_temp_c'])
        
        insights.append(f"Hottest day will be {hottest_day['day_name']} with a max temperature of {hottest_day['max_temp_c']}°C")
        insights.append(f"Coolest day will be {coolest_day['day_name']} with a min temperature of {coolest_day['min_temp_c']}°C")
        
        # Find average values
        avg_temp = sum(day['avg_temp_c'] for day in forecast) / len(forecast)
        avg_humidity = sum(day['avg_humidity'] for day in forecast) / len(forecast)
        
        insights.append(f"Average temperature over the next 7 days: {avg_temp:.1f}°C")
        insights.append(f"Average humidity over the next 7 days: {avg_humidity:.1f}%")
        
        # Current weather insight
        insights.append(f"Current temperature in {location['name']}: {current['temp_c']}°C (feels like {current['feelslike_c']}°C)")
        insights.append(f"Current condition: {current['condition']}")
        
        return render_template(
            "prediction.html",
            temp_graph=temp_graph,
            hum_graph=hum_graph,
            rain_graph=rain_graph,
            hourly_temp_graph=hourly_temp_graph,
            current=current,
            forecast=forecast,
            location=location,
            insights=insights
        )
        
    except Exception as e:
        return render_template("prediction.html", error=f"Error in prediction module: {str(e)}")

# ---------------------------
# historical route
# ---------------------------
@app.route("/historical")
def historical():
    try:
        df = load_and_prepare()
    except Exception as e:
        return f"Error loading data: {str(e)}"

    # Descriptive stats table
    stats_df = descriptive_stats(df)
    stats_html = stats_df.to_html(index=False, classes="table table-sm table-bordered")

    # Extremes
    extremes = detect_extremes(df)

    # Narrative
    narrative = narrative_insights(df, stats_df, extremes)
    
    # Generate insights list
    insights_list = generate_insights_list(df, extremes)

    # Summary insight list (for top panel)
    summary_insights = []
    # dataset coverage
    years = df["Year"].dropna().astype(int).astype(str).tolist()
    if years:
        summary_insights.append(f"Dataset covers years {years[0]} to {years[-1]}.")
    # extremes
    if extremes.get("MaxRainfallYear"):
        summary_insights.append(f"Peak rainfall in {extremes['MaxRainfallYear']}: {extremes['MaxRainfallValue']:.1f} mm.")
    if extremes.get("MaxTempYear"):
        summary_insights.append(f"Hottest year recorded: {extremes['MaxTempYear']} at {extremes['MaxTempValue']:.1f}°C.")
    if extremes.get("FloodYears"):
        summary_insights.append(f"Flood events recorded in {len(extremes['FloodYears'])} years.")

    # Graphs (metrics)
    metric_info = [
        ("Temperature", "Average Temperature (°C)"),
        ("Rainfall", "Annual Rainfall (mm)"),
        ("Humidity", "Relative Humidity (%)"),
        ("CO2", "CO₂ Emissions (kt)"),
        ("WindSpeed", "Wind Speed (kmph)")
    ]
    graphs = []
    for col, title in metric_info:
        html = plot_metric(df, col, title)
        if html:
            graphs.append({"col": col, "title": title, "html": html})

    # Correlation heatmap
    numeric_df = df[["Temperature","Rainfall","Humidity","CO2","WindSpeed"]].select_dtypes(include=[np.number])
    corr_html = None
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        corr_fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        corr_html = corr_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Event and Impact Table
    events_html = None
    if "Event_Notes" in df.columns and "Impact_Observed" in df.columns:
        events_df = df[["Year", "Event_Notes", "Impact_Observed"]].copy()
        # Filter out rows with no events or impacts
        events_df = events_df[(events_df["Event_Notes"] != "") | (events_df["Impact_Observed"] != "")]
        if not events_df.empty:
            events_html = events_df.to_html(index=False, classes="table table-sm table-bordered")

    return render_template(
        "historical.html",
        stats_html=stats_html,
        summary_insights=summary_insights,
        graphs=graphs,
        corr_html=corr_html,
        narrative=narrative,
        extremes=extremes,
        insights_list=insights_list,
        events_html=events_html
    )

# -----------------------------
# FeaturePro (interactive graphs & insights)
# -----------------------------
@app.route("/featurepro", methods=["GET", "POST"], endpoint='featurepro')
def feature_pro():
    try:
        # Load data for FeaturePro
        df = load_and_prepare()
    except Exception as e:
        return render_template("featurepro.html", 
                              error=f"❌ Error loading data: {e}",
                              parameters=[],
                              years=[],
                              graphs=[],
                              insights=[],
                              selected_params=[],
                              year_start=None,
                              year_end=None)

    # Parameters available for selection
    available_params = [
        c for c in [
            "Temperature", "Rainfall", "Humidity", "CO2", "WindSpeed"
        ] if c in df.columns
    ]

    # Ensure Year column exists
    if "Year" not in df.columns:
        return render_template("featurepro.html", 
                              error="❌ 'Year' column missing in dataset.",
                              parameters=[],
                              years=[],
                              graphs=[],
                              insights=[],
                              selected_params=[],
                              year_start=None,
                              year_end=None)

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        return render_template("featurepro.html", 
                              error="❌ No valid years found in dataset.",
                              parameters=[],
                              years=[],
                              graphs=[],
                              insights=[],
                              selected_params=[],
                              year_start=None,
                              year_end=None)

    # Default selection
    selected_params = available_params.copy()
    year_start = int(years[0])
    year_end = int(years[-1])

    graphs = []
    insights = []

    # Handle POST (user input)
    if request.method == "POST":
        form_params = request.form.getlist("parameters")
        selected_params = form_params if form_params else selected_params

        try:
            year_start = int(request.form.get("year_start") or year_start)
            year_end = int(request.form.get("year_end") or year_end)
        except ValueError:
            pass

    # Validate year range
    if year_start > year_end:
        year_start, year_end = year_end, year_start

    # Filter dataset for year range
    df_filtered = df[(df["Year"] >= year_start) & (df["Year"] <= year_end)]

    # Create graphs
    for param in selected_params:
        if param not in df_filtered.columns or df_filtered[param].dropna().empty:
            insights.append(f"⚠ Insufficient data for {param}")
            continue

        df_plot = (
            df_filtered[["Year", param]]
            .dropna(subset=["Year", param])
            .sort_values("Year")
        )

        if df_plot.empty:
            insights.append(f"⚠ No data available for {param} in selected time range")
            continue

        fig = px.line(
            df_plot,
            x="Year",
            y=param,
            title=f"{param} ({year_start} - {year_end})",
            markers=True
        )
        
        # Store both title and HTML
        graphs.append({
            'title': f"{param} ({year_start} - {year_end})",
            'html': fig.to_html(full_html=False, include_plotlyjs="cdn")
        })

        # Generate insight
        try:
            change = df_plot[param].iloc[-1] - df_plot[param].iloc[0]
            perc = (
                (change / df_plot[param].iloc[0]) * 100
                if df_plot[param].iloc[0] != 0
                else 0
            )
            direction = "increased" if change > 0 else "decreased"
            arrow = "⬆" if change > 0 else "⬇"
            insights.append(
                f"{arrow} {param} {direction} by {abs(change):.2f} ({abs(perc):.1f}%)"
            )
        except Exception:
            insights.append(f"⚠ Insight not available for {param}")

    return render_template(
        "featurepro.html",
        parameters=available_params,
        years=years,
        graphs=graphs,
        insights=insights,
        selected_params=selected_params,
        year_start=year_start,
        year_end=year_end,
        error=None
    )

# ---------------------------
# Weather route
# ---------------------------
@app.route("/weather")
def weather():
    """Render the weather checker page"""
    return render_template("weather.html")

# ---------------------------
# IoT route
# ---------------------------
@app.route("/iot")
def iot():
    """Render the IoT integration page"""
    return render_template("iot.html")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # simple sanity-run checks (no crash on startup)
    try:
        df_test = load_and_prepare()
        print("Loaded data. Years:", df_test["Year"].dropna().astype(int).unique().tolist()[:5], "...")
    except Exception as e:
        print("Startup data load error:", e)
    app.run(debug=True)