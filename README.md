# diff-privacy-OpenDP
Prise en main d'OpenDP dans le cadre d'un stage 3A

Package à installer :
- 'opendp[polars]'

Lancer example.py si vous voulez voir une idée de où ca en est (Le main.py n'est pas à jour encore).

----

Sous réserve que les requêtes avec filtre fonctionne, le filtre et le group_by n'impacte pas les bruits injectés
Sensibilité en max(abs(L), U) est bien dépendant de la moyenne, somme.

Deux cas de figure : 
- on raisonne en terme de unbounded DP donc de type datasize inconnu.
- on raisonne en terme de bounded DP donc datasize connu => dp.polars.Margin(public_info="lengths") 
    ce qui permet d'avoir une sensibilité en (d_in // 2) * (U - L) où d_in contribution d'un individu. 
    Seulement visiblement d_in > 1 (notion de distance en terme d'ajout/suppression de lignes ?) donc augmentation de la sensibilité pour les autres requetes
    + Tout est cassé en cas de group_by (fixable) ou de filter (plus difficilement)

Il est équivalent de faire une requete count et une requete somme, que faire une requete moyenne (si on raisonne en terme unbounded dp = taille dataset peut changer).
Si on veut accorder des budgets différents, cela peut avoir son importance.

Pour les quantiles, le scale dépend de l'ordre du quantile. Il est équivalent de faire 3 requetes demandant un unique quantile et 1 requete demandant les 3 quantiles
directement (car elle divise équitablement le budget entre les quantiles). Cela peut avoir son importance si on veut justement diviser de manière non équitable.

A faire : ajouter un nouveau mécanisme pour la moyenne
map(d_in) = d_in // 2 * (U - L) / n (option resize non disponible ici)
Explorer make_private_lazyframe


Impossible de mettre une marge lengths connu si privacy unit=1