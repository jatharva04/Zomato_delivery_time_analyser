import requests

data = {
    "Type_of_vehicle":"Scooter",
    "Weather_conditions":"Windy",
    "Road_traffic_density":"Jam",
    "Festival":"Yes", 
    "City":"Metropolitan",
    "distance (km)":"6.24"
}

response = requests.post("http://localhost:8080/predict", json=data)
print(response.json())
