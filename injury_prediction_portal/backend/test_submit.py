import requests

url = "http://127.0.0.1:5000/submit"

data = {
    "Name": "Athlete X",
    "Age": 24,
    "Gender": "Male",
    "Sport": "Football",
    "Injury Duration (weeks)": 2,
    "Injury Occurred (weeks ago)": 3,
    "Trunk Flexion (cm)": 15,
    "BMI": 22.5,
    "Weekly Training Hours": 10,
    "Current discomfort / Injury": "Yes",
    "Shoulder Flexion (deg)": 170,
    "Weight (kg)": 70,
    "Previous Injuries": 2,
    "Position": "Midfielder",
    "Coach": "John Doe",
    "Coach exp": 5,
    "coaches success %": 80
}

response = requests.post(url, json=data)

print("STATUS:", response.status_code)
print("RESPONSE:", response.json())
