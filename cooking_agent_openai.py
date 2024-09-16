import openai
import requests

# Clé API OpenAI (remplace ceci par ta clé API secrète)
openai.api_key = 'OPENAI_API_KEY'

# URL de l'API wttr.in
WEATHER_API_URL = "https://wttr.in/{city}?format=%C"

# Fonction pour récupérer les conditions météorologiques
def get_weather(city):
    response = requests.get(WEATHER_API_URL.format(city=city))
    if response.status_code == 200:
        weather_info = response.text.strip()
        return weather_info
    else:
        return "Unable to retrieve weather data."

# Fonction pour générer des idées de recettes en fonction de la météo
def generate_recipe(condition):
    prompt = f"Suggest a recipe for {condition} weather."

    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    recipe = response.choices[0].message['content'].strip()
    return recipe

# Exemple d'utilisation
if __name__ == "__main__":
    city_input = input("Enter the city name: ")
    condition = get_weather(city_input)
    print(f"Weather condition in {city_input}: {condition}")
    
    if condition != "Unable to retrieve weather data.":
        recipe = generate_recipe(condition)
        print(f"Suggested recipe: {recipe}")
    else:
        print("Could not retrieve weather information.")
