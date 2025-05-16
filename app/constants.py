import numpy as np
import polars as pl
import opendp.prelude as dp

dp.enable_features("contrib")

# Simuler les données une seule fois, stockées dans une variable globale
np.random.seed(42)

N = 1000
regions_fr = [
    "Île-de-France", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine", "Occitanie",
    "Provence-Alpes-Côte d'Azur", "Grand Est", "Hauts-de-France", "Bretagne",
    "Normandie", "Centre-Val de Loire", "Pays de la Loire", "Bourgogne-Franche-Comté",
    "Corse"
]

DF_POLARS = pl.DataFrame({
    "sexe": np.random.choice(["H", "F"], size=N),
    "region": np.random.choice(regions_fr, size=N),
    "revenu_annuel": np.random.randint(10000, 80000, size=N),
    "profession": np.random.choice(["ingénieur", "médecin", "avocat"], size=N),
    "note_satisfaction": np.random.randint(0, 20, size=N),
    "heures_travaillees": np.random.randint(20, 60, size=N),
    "secteur d'activité": np.random.choice(["public", "privé", "associatif"], size=N)
})

# Clés valides pour les filtres
KEY_VALUES = {
    "sexe": ["H", "F"],
    "region": regions_fr,
    "profession": ["ingénieur", "médecin", "avocat"],
    "secteur d'activité": ["public", "privé", "associatif"]
}

# Contexte DP, construit une seule fois
CONTEXT_PARAM = {
    "data": DF_POLARS.lazy(),
    "privacy_unit": dp.unit_of(contributions=1),
    "margins": [
        dp.polars.Margin(max_partition_length=1000),
        dp.polars.Margin(
            by=["region"],
            public_info="lengths",
            max_partition_length=1000
        ),
        dp.polars.Margin(
            by=["secteur d'activité", "region", "sexe", "profession"],
            public_info="keys",
            max_partition_length=1000
        ),
    ],
}
