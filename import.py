import os

# 1. Créer le dossier 'data' s'il n'existe pas
os.makedirs("data", exist_ok=True)

# 2. Télécharger le fichier si besoin
dataset_url = "https://raw.githubusercontent.com/opendp/opendp/main/docs/source/data/PUMS_california_demographics_1000/data.csv"
dataset_path = "data/data.csv"

if not os.path.exists(dataset_path):
    import urllib.request
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download complete!")
