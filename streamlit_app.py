import streamlit as st

st.set_page_config(page_title="Diff Privacy App", page_icon="🔐", layout="wide")

page_1 = st.Page("app/page_1.py", title="Dataset", icon=":material/add_circle:")
page_2 = st.Page("app/page_2.py", title="Requêtes", icon="🔥")

page_3 = st.Page("app/page_3.py", title="Bruit Gaussien", icon="🔥")
page_4 = st.Page("app/page_4.py", title="Gestion du budget", icon=":material/delete:")
page_5 = st.Page("app/page_5.py", title="Résultat DP", icon="🔥")

pg = st.navigation([page_1, page_2])
pg = st.navigation(
        {
            "Données en entrée": [page_1, page_2],
            "Mécanisme": [page_3],
            "Differential Privacy": [page_4, page_5]
        }
    )
pg.run()
