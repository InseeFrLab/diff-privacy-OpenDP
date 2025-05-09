import json
import polars as pl
import opendp.prelude as dp
import numpy as np
from fonctions import process_request, process_request_dp, rho_from_eps_delta, eps_from_rho_delta

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

context_param = {
    "data": df.lazy(),
    "privacy_unit": dp.unit_of(contributions=1),
    "margins": [
        dp.polars.Margin(max_partition_length=1000),
        dp.polars.Margin(
            by=["secteur d'activité", "region", "sexe", "profession"],
            public_info="keys",
        ),
    ],
}

nb_req = len(requetes)
print("Le nombre de requêtes a traité est de", nb_req)

eps_tot = 3
delta_tot = 1e-5
print(f"Le budget fixé pour l'étude est le suivant : ({eps_tot}, {delta_tot})")

rho_tot = rho_from_eps_delta(eps_tot, delta_tot)

nb_req_rho = sum(
    1 for req in requetes.values() if req.get("type") != "quantile"
)
print("Le nombre de requêtes pouvant utilisé du bruit gaussien est de", nb_req_rho)

budget_rho = dp.loss_of(rho=rho_tot * nb_req_rho / nb_req)
eps_depense = eps_from_rho_delta(rho_tot * nb_req_rho / nb_req, delta_tot)
print(f"Le budget depensé par les requêtes utilisant du bruit gaussien est ({eps_depense}, {delta_tot})")

budget_eps_restant = dp.loss_of(epsilon=eps_tot - eps_depense)
print(f"Tandis que le budget depensé pour les requêtes demandant un ou plusieurs quantiles est de {eps_tot - eps_depense}")

context_rho = dp.Context.compositor(
    **context_param,
    privacy_loss=budget_rho,
    split_evenly_over=nb_req_rho
)

context_eps = dp.Context.compositor(
    **context_param,
    privacy_loss=budget_eps_restant,
    split_evenly_over=nb_req - nb_req_rho
)

# Application du traitement à chaque requête
for key, req in requetes.items():
    # print(f"\n Traitement de : {key}")
    # resultat = process_request(df.lazy(), req)
    # print(resultat)

    print(f"\n🔍 Traitement DP de : {key}")
    resultat = process_request_dp(context_rho, context_eps, key_values, req)
    print(resultat.summarize())
    print(resultat.release().collect())
