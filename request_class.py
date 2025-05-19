from abc import ABC, abstractmethod
import opendp.prelude as dp
import polars as pl
from fonctions import parse_filter_string


# Classe mère
class request_dp(ABC):
    def __init__(self, context, key_values, by=None, variable=None, bounds=None, filtre=None):
        self.context = context
        self.key_values = key_values
        self.by = by
        self.variable = variable
        self.bounds = bounds
        self.filtre = filtre

    def generate_public_keys(self, by_keys: list[str]) -> pl.LazyFrame:
        from itertools import product

        # Ne garder que les colonnes utiles pour le group_by
        values = [self.key_values[key] for key in by_keys if key in self.key_values]

        # Produit cartésien des valeurs
        combinaisons = list(product(*values))

        # Construction du LazyFrame public
        public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
        return public_keys

    @abstractmethod
    def execute(self):
        pass


class count_dp(request_dp):
    """
    Comptage différentiellement privé – Notes de la fonction d'OpenDP

    - Sensibilité = privacy_unit

    - Ne pas exécuter la requête si la marge est supposée connue mais non fournie
        → sinon : ValueError: unable to infer bounds
        → la contourner via une requête jointe ne fonctionne pas ici

    - La requête peut se faire en (ε)-DP (Laplace) ou (ρ)-DP (Gaussien)

    - Si group_by :
        - Les clés doivent être incluses dans une margin sinon fallback en (ε, δ)-DP avec seuil
        - ε affecte l'écart-type du bruit + le seuil
        - δ affecte uniquement le seuil
        - Formule classique pour l'écart-type, mais le seuil reste mal compris

    - Si filter :
        - La margin n’est pas prise en compte
        - Problème potentiel avec group_by
        - Solution : jointure avec un ensemble de clés (complet, partiel ou vide)
        - Si budget en (ε, δ), aucun seuil n’est appliqué
    """
    """
 
        - Ne pas faire la requête si on suppose la marge connue sinon erreur 'ValueError: unable to infer bounds'
            -> Solution = la faire en même temps qu'une autre requête (marche pas avec ma fonction)
        - Ne pas faire la requête si on suppose la marge connue
        - Requête possible via epsilon (Laplace) ou rho (Gaussian) selon les formules définies

        - Si group_by : a mettre les clés de la variables dans un margin sinon requête en (epsilon, delta) via un seuil
            - changer epsilon bouge l'écart type du bruit et le seuil
            - changer delta modifie uniquement le seuil
            - l'écart type correspond à la formule classique mais pas d'idée pour le seuil

        - Si filter : la marge n'est pas prise en compte => problème avec group_by
            Solution = faire une jointure avec un set de clés (peut contenir tout, partiellement, voire aucunes des véritables clés de la variable)
            Se le budget était en (epsilon, delta) alors pas de seuil
    """
    def execute(self):
        query = self.context.query()

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(dp.len().alias("count"))
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(dp.len().alias("count"))

        return query


class sum_dp(request_dp):
    """
    Somme :
        - Sensibilité = à la privacy_unit * max(abs(L), abs(U)) ou (privacy_unit // 2) * (U - L) si marge variable connu
        - Requête possible via epsilon (Laplace) ou rho (Gaussian) selon les formules définies
        - Obligé de définir un dp.polars.Margin(max_partition_length=?)

        - Si group_by : a mettre les clés de la variables dans un margin sinon erreur
            Autre solution : on effectue en même temps une requete de comptage en (epsilon, delta) via le seuil
            Autre solution : jointure

        - Si filter : la marge n'est pas prise en compte => sensibilité (formule avec max) et problème avec group_by
            Solution (partielle car pas sensibilité) = faire une jointure avec un set de clés 
                (peut contenir tout, partiellement, voire aucunes des véritables clés de la variable)

        Attention privacy unit=1 si marge connu sur le dataset et seulement les clés connus (mais pas les marges dessus) sinon erreur

        - Si entrée type float :
            - Si privacy unit = 1 -> problème dans le scale qui est ressort un résultat dépendant de max_partition_length, de valeur très faible sauf dans le cas au dessus
            - Si group_by :
                    OpenDPException:
                        MakeTransformation("max_num_partitions must be known when the metric is not sensitive to ordering (SymmetricDistance)")
                        Predicate in binary search always raises an exception. This exception is raised when the predicate is evaluated at 0.0.

                Il faut forcement définir max_num_partitions dans le bon margin sauf si on faire la jointure après
    """
    def execute(self):
        l, u = self.bounds
        query = self.context.query()
        expr = (
            pl.col(self.variable)
            .fill_null(0)  # .fill_nan(0) à ajouter si float (et seulement si ...)
            .dp.sum((l, u))
            .alias("sum")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        return query


class mean_dp(request_dp):
    """
    Moyenne :
        - Sensibilité = cas de la somme et du comptage
        - Requête possible via epsilon (Laplace) ou rho (Gaussian) selon les formules définies
        - Obligé de définir un dp.polars.Margin(max_partition_length=?)

        - Si group_by : a mettre les clés de la variables dans un margin sinon erreur
            Attention : on effectue en même temps une requete de comptage en (epsilon, delta) via le seuil ne MARCHE PAS
            Autre solution : jointure

        - Si filter : la marge n'est pas prise en compte => sensibilité (formule avec max) et problème avec group_by
            Solution = faire une jointure avec un set de clés (peut contenir tout, partiellement, voire aucunes des véritables clés de la variable)

        Attention privacy unit=1 si marge connu sur le dataset et seulement les clés connus (mais pas les marges dessus) sinon erreur

        - Si entrée type float :
            - Si privacy unit = 1 -> problème dans le scale qui est ressort un résultat dépendant de max_partition_length, de valeur très faible sauf dans le cas au dessus
            - Si group_by :
                    OpenDPException:
                        MakeTransformation("max_num_partitions must be known when the metric is not sensitive to ordering (SymmetricDistance)")
                        Predicate in binary search always raises an exception. This exception is raised when the predicate is evaluated at 0.0.

                Il faut forcement définir max_num_partitions dans le bon margin sauf si on faire la jointure après
    """
    def execute(self):
        l, u = self.bounds
        query = self.context.query()
        expr = (
            pl.col(self.variable)
            .fill_null(0)  # .fill_nan(0) à ajouter si float (et seulement si ...)
            .dp.mean(bounds=(l, u))
            .alias("mean")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        return query


class quantile_dp(request_dp):
    """
    Quantile :
        - Sensibilité proportionnelle à la privacy_unit
        - Requête possible via epsilon (Laplace) seulement
        - Obligé de définir un dp.polars.Margin(max_partition_length=?)

        - Si marge dataset connu, impossible privacy unit = 1 et la sensibilité proportionnelle à (privacy_unit // 2) * 4 * sensibilité de base (conjecture)

        - Si group_by :
            - Mettre les clés de la variables dans un margin sinon requête en (epsilon, delta) via un seuil ou bien jointure
            - Plus on augmente privacy unit, plus la sensibilité augmente rapidement (2, 8, 18, 32, 40, 48, 56) indépendamment des marges

        - Si ordre du quantile différent de 0, 0.5 et 1 alors résultat apparement très bruité

        - Si filter : la marge n'est pas prise en compte => problème avec group_by
            Solution = faire une jointure avec un set de clés (peut contenir tout, partiellement, voire aucunes des véritables clés de la variable)
            Se le budget était en (epsilon, delta) alors pas de seuil

        Dans le meilleur des cas, scale des quartiles (0, 0.25, 0.5, 0.75, 1) en (2, 6, 2, 6, 2).
        Le reste c'est trop bruité
    """
    def __init__(self, context, key_values, variable, candidats, by=None, bounds=None, filtre=None, alpha=None):
        super().__init__(context, key_values, by, variable, bounds, filtre)

        # Attention, il y a y un problème avec OpenDP au niveau des candidats
        # Il faut qu'ils soient ordonnées de manière croissante ET de partie entière distincte
        if candidats["type"] == "list":
            self.candidats = candidats["values"]
        elif candidats["type"] == "range":
            import numpy as np
            self.candidats = np.arange(candidats["min"], candidats["max"] + candidats["step"], candidats["step"]).tolist()

        if alpha is None:
            self.list_alpha = [0.5]
        elif isinstance(alpha, list):
            self.list_alpha = alpha
        else:
            self.list_alpha = [alpha]

    def execute(self):
        query = self.context.query()
        aggs = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(a, self.candidats)
            .alias(f"quantile_{a}")
            for a in self.list_alpha
        ]

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(*aggs)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(*aggs)

        return query
