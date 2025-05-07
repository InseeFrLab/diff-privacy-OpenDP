import requests
import json
import os
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env
load_dotenv()

# Récupère le token depuis la variable d'environnement
token = os.getenv("TOKEN")

def extract_information_from_text(token, texte):
    url = 'https://llm.lab.sspcloud.fr/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Prompt enrichi avec le paramètre filtre
    prompt = f"""
    Tu vas extraire les paramètres de plusieurs requêtes à partir du texte suivant :
    {texte}

    Il peut s'agir de requêtes de type :
    - comptage
    - moyenne
    - quantile (avec alpha)
    - somme

    Pour chaque requête, retourne un objet JSON structuré avec :
    - type : "count", "mean", "quantile" ou "sum"
    - variable : la variable concernée
    - bounds : les bornes sous forme de tuple (L, U), ou null
    - by : liste des variables de regroupement, ou null
    - alpha : pour les quantiles uniquement (sinon null)
    - filtre : chaîne représentant un filtre logique (par exemple : "var_1 > 0 & var_2 == 3 | var_4 != 'toto'"), ou null s'il n'y a pas de condition

    La sortie attendue est un dictionnaire avec des clés de la forme "req_1", "req_2", etc.
    Réponds uniquement avec un JSON **valide** (sans texte autour), et sans commentaires.

    Si une information est absente, mets `null`.
    """

    data = {
        "model": "llama3.3:70b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def extract_information(response_json):
    try:
        content = response_json['choices'][0]['message']['content']
        json_string = content.strip().strip('```json').strip('```').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Erreur d'extraction : {e}")
        return None


# Exemple de texte enrichi avec des filtres
texte = """
Nous avons besoin de lancer les analyses suivantes :
1. Un comptage de la variable 'sexe' regroupée par 'region', uniquement pour les personnes dont "age" est de plus de 18 ans.
2. La moyenne de 'revenu_annuel' dans les bornes (10000, 75000), regroupée par 'profession' et 'region', uniquement si 'statut' est égal à 'actif'.
3. La somme de la variable 'heures_travaillees' dans l'intervalle (30, 60) par 'profession', pour les employés ayant 'contrat' différent de 'CDD' ou travaillant dans 'secteur' = 'industrie'.
4. Calcul du quantile d'ordre 0.5 sur 'note_satisfaction' regroupé par 'profession', pour les personnes dont 'note_confiance' est supérieure à 50.
"""

# Traitement
resultat = extract_information_from_text(token, texte)
extracted_data = extract_information(resultat)

# Affichage
print(json.dumps(extracted_data, indent=4, ensure_ascii=False))

# S'assurer que le dossier "data" existe
os.makedirs("data", exist_ok=True)

# Chemin du fichier de sortie
output_path = "data/data.json"

# Sauvegarde du JSON dans un fichier
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, indent=4, ensure_ascii=False)

print(f"Fichier enregistré sous : {output_path}")
