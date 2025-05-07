import json
import polars as pl
import opendp.prelude as dp
import numpy as np
from fonctions import process_request, process_request_dp


dp.enable_features("contrib")

# Charger le JSON enregistré
with open("data_example.json", encoding="utf-8") as f:
    requetes = json.load(f)

n = 1000
np.random.seed(42)  # pour reproductibilité

df = pl.DataFrame({
    "sexe": np.random.choice(["H", "F"], size=n),
    "region": np.random.choice(["Nord", "Sud"], size=n),
    "revenu_annuel": np.random.randint(10000, 80000, size=n),
    "profession": np.random.choice(["ingénieur", "médecin", "avocat"], size=n),
    "note_satisfaction": np.random.randint(0, 20, size=n),
    "heures_travaillees": np.random.randint(20, 60, size=n),
    "secteur d'activité": np.random.choice(["public", "privé", "associatif"], size=n)
})

# Définir les valeurs possibles pour chaque clé
key_values = {
    "sexe": ["H", "F"],
    "region": ["Nord", "Sud"],
    "profession": ["ingénieur", "médecin", "avocat"],
    "secteur d'activité": ["public", "privé", "associatif"]
}

ind_contrib = dp.unit_of(contributions=1)
budget = dp.loss_of(epsilon=1.)
nb_requetes = len(requetes)

context = dp.Context.compositor(
    data=df.lazy(),
    privacy_unit=ind_contrib,
    privacy_loss=budget,
    split_evenly_over=nb_requetes,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than France population
            max_partition_length=1000
        ),
        dp.polars.Margin(
            by=["secteur d'activité", "region", "sexe", "profession"],
            public_info="keys",
        ),
    ],
)

# Application du traitement à chaque requête
for key, req in requetes.items():
    # print(f"\n Traitement de : {key}")
    # resultat = process_request(df.lazy(), req)
    # print(resultat)

    print(f"\n🔍 Traitement DP de : {key}")
    resultat = process_request_dp(context, key_values, req)
    print(resultat.summarize())
    print(resultat.release().collect())
