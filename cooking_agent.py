import pandas as pd
import requests
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Constants
WEATHER_API_URL = "https://wttr.in/{city}?format=%C+%t"


# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5.0 / 9.0


# Function to fetch weather data for a city
def get_weather(city):
    response = requests.get(WEATHER_API_URL.format(city=city))
    if response.status_code == 200:
        weather_info = response.text.strip()
        condition, temperature = weather_info.split(' ', 1)
        temp_match = re.search(r'([-+]?\d*\.?\d+)', temperature)
        temp_value = float(temp_match.group(0)) if temp_match else None
        # Convert temperature if in Fahrenheit
        if 'F' in temperature:
            temp_value = fahrenheit_to_celsius(temp_value)
        return condition, temp_value
    return "Unable to retrieve weather data.", None


# Function to predict recipe based on weather data
def predict_recipe(city):
    condition, temperature = get_weather(city)
    if temperature is None:
        return condition

    # Determine weather category
    if "rain" in condition.lower() or "drizzle" in condition.lower() or "shower" in condition.lower():
        category = "rainy"
    elif "clear" in condition.lower() or "sunny" in condition.lower():
        category = "hot" if temperature > 25 else "clear"
    elif "snow" in condition.lower() or temperature < 5:
        category = "snowy"
    elif temperature < 15:
        category = "cold"
    elif "windy" in condition.lower():
        category = "windy"
    else:
        category = "clear"

    # Prepare new data for prediction
    new_data_df = pd.DataFrame({
        'temperature': [temperature],
        'weather_condition': [category]
    })

    # Encode new data
    new_weather_encoded = encoder.transform(new_data_df[['weather_condition']])
    new_scaled_features = scaler.transform(new_data_df[['temperature']])
    new_X = pd.concat([pd.DataFrame(new_scaled_features), pd.DataFrame(new_weather_encoded)], axis=1)

    # Predict recipe
    predictions = model.predict_proba(new_X)

    # Get the top 5 recipe suggestions
    recipe_indices = predictions[0].argsort()[-5:][::-1]
    recipe_names = model.classes_[recipe_indices]

    # Return the recipe with the highest probability
    return recipe_names[0]


# Load data
data = pd.read_csv('recipes_10000.csv')

# Encode categorical variables (weather_condition)
encoder = OneHotEncoder(sparse_output=False)
weather_encoded = encoder.fit_transform(data[['weather_condition']])

# Normalize numeric variables (temperature)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['temperature']])

# Combine encoded and normalized features
X = pd.concat([pd.DataFrame(scaled_features), pd.DataFrame(weather_encoded)], axis=1)

# Target variable (recipes)
y = data['recipe_name']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Example usage
if __name__ == "__main__":
    city_input = input("Enter the city name: ")
    print(f"For {city_input}, we recommend the recipe: {predict_recipe(city_input)}")
