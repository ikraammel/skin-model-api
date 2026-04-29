import requests

url = "http://127.0.0.1:5000/predict"

# Remplace par le chemin d'une vraie image sur ton PC
image_path = "test.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"image": f})

result = response.json()

print(f"\n Prédiction : {result['prediction']}")
print(f" Confiance  : {result['confidence']}")
print(f"\n Toutes les probabilités :")
for classe, prob in result['all_probabilities'].items():
    bar = '█' * int(prob / 5)
    print(f"  {classe:<15} {prob:5.2f}%  {bar}")