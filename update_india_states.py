import json

# List of 33 states from model_metadata.json
training_states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "DNH", "Delhi", "Goa", "Gujarat", "HP", "Haryana", "J&K",
    "Jharkhand", "Karnataka", "Kerala", "MP", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Pondy", "Odisha", "Punjab", "Rajasthan", "Tripura", "Sikkim",
    "Tamil Nadu", "Telangana", "West Bengal", "Uttarakhand", "UP", "Maharashtra"
]

# Mapping from JSON NAME_1 to training state names
state_mapping = {
    'AndamanandNicobar': None,
    'AndhraPradesh': 'Andhra Pradesh',
    'ArunachalPradesh': 'Arunachal Pradesh',
    'Assam': 'Assam',
    'Bihar': 'Bihar',
    'Chandigarh': 'Chandigarh',
    'Chhattisgarh': 'Chhattisgarh',
    'DadraandNagarHaveli': 'DNH',
    'DamanandDiu': None,
    'Goa': 'Goa',
    'Gujarat': 'Gujarat',
    'Haryana': 'Haryana',
    'HimachalPradesh': 'HP',
    'JammuandKashmir': 'J&K',
    'Jharkhand': 'Jharkhand',
    'Karnataka': 'Karnataka',
    'Kerala': 'Kerala',
    'Lakshadweep': None,
    'MadhyaPradesh': 'MP',
    'Maharashtra': 'Maharashtra',
    'Manipur': 'Manipur',
    'Meghalaya': 'Meghalaya',
    'Mizoram': 'Mizoram',
    'Nagaland': 'Nagaland',
    'NCTofDelhi': 'Delhi',
    'Odisha': 'Odisha',
    'Puducherry': 'Pondy',
    'Punjab': 'Punjab',
    'Rajasthan': 'Rajasthan',
    'Sikkim': 'Sikkim',
    'TamilNadu': 'Tamil Nadu',
    'Telangana': 'Telangana',
    'Tripura': 'Tripura',
    'UttarPradesh': 'UP',
    'Uttarakhand': 'Uttarakhand',
    'WestBengal': 'West Bengal'
}

# Load JSON
with open('india_states.json', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

# Update NAME_1
for feature in geojson['features']:
    name_1 = feature['properties']['NAME_1']
    if name_1 in state_mapping:
        new_name = state_mapping[name_1]
        if new_name in training_states:
            feature['properties']['NAME_1'] = new_name
        else:
            feature['properties']['NAME_1'] = None
            print(f"Warning: {name_1} mapped to {new_name} (not in training states)")
    else:
        feature['properties']['NAME_1'] = None
        print(f"Warning: No mapping for {name_1}")

# Save updated JSON
with open('india_states_updated.json', 'w', encoding='utf-8') as f:
    json.dump(geojson, f, indent=4)
print("Updated JSON saved as 'india_states_updated.json'")