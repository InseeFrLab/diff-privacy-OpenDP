import json
import polars as pl
import opendp.prelude as dp
from fonction import *
import numpy as np
import os
import s3fs
import pandas as pd
import geopandas as gpd

dp.enable_features("contrib")

BUCKET = "stuartbenoliel"
FILE_KEY_S3 = "stage_3A/population_974.gpkg"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://' + 'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"])

# Charger le fichier .gpkg
with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    df = gpd.read_file(file_in)

df = pl.DataFrame(df.drop(columns="geometry"))

print(df.head())

# Charger le JSON enregistré
with open("data/request_reunion.json", encoding="utf-8") as f:
    requetes = json.load(f)


nb_requetes = len(requetes)

context = dp.Context.compositor(
    data=df.lazy(),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=5.0),
    split_evenly_over=nb_requetes,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than
            #    France population
            public_info="lengths",  # make partition size public (bounded-DP)
            max_partition_length=60_000_000
        ),
        dp.polars.Margin(
            by=["HOUSEHOLD_SIZE", "AGE_CAT", "STATUT"],
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
