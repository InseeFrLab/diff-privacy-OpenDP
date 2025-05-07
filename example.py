import json
import polars as pl
import opendp.prelude as dp
from fonctions import *
import numpy as np


dp.enable_features("contrib")

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

# Charger le JSON enregistré
with open("data/data.json", encoding="utf-8") as f:
    requetes = json.load(f)


nb_requetes = len(requetes)

context = dp.Context.compositor(
    data=df.lazy(),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=5.0),
    split_evenly_over=4,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than
            #    France population
            public_info="lengths",  # make partition size public (bounded-DP)
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
    print(f"\n🔍 Traitement de : {key}")
    resultat = process_request(df.lazy(), req)
    print(resultat)

    print(f"\n🔍 DP Traitement de : {key}")
    resultat = process_request_dp(context, req)
    print(resultat)
