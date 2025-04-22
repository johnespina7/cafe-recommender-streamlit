
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar la base de datos desde Google Sheets
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/11-LCzX3i9zlnUoqBor8tTZHwpge8hJ6W0dmeC8Rb734/gviz/tq?tqx=out:csv&sheet=Hoja1"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("✨ Tu Café Ideal - El Mundo del Café")
st.markdown("Responde brevemente y descubre un café diseñado para tu paladar.")

# Preguntas al usuario
notas_usuario = st.multiselect("¿Qué notas sensoriales te atraen más?", 
                               ["floral", "ácido", "cítrico", "chocolate", "caramelo", "nuez", "frutal", "té negro", "jazmín", "especiado"])
intensidad = st.selectbox("¿Qué intensidad prefieres?", [1, 2, 3])
emocion = st.selectbox("¿Qué estilo emocional te representa más hoy?", ["Sutil", "Refrescante", "Intenso", "Confortable", "Ligero", "Equilibrado", "Cálido", "Vibrante"])

if st.button("Descubrir mi café ideal"):
    # Procesamiento de texto
    df["Notas_Procesadas"] = df["Notas Sensoriales"].str.lower().str.replace(",", " ")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Notas_Procesadas"])

    user_profile_text = " ".join(notas_usuario)
    user_vec = vectorizer.transform([user_profile_text])
    similarities = cosine_similarity(user_vec, X).flatten()
    df["Puntaje Afinidad"] = similarities

    # Filtrar por intensidad y emoción
    filtro = (df["Intensidad"] == intensidad) & (df["Perfil Emocional"].str.lower() == emocion.lower())
    recomendados = df[filtro].sort_values(by="Puntaje Afinidad", ascending=False).head(3)

    if recomendados.empty:
        st.warning("No encontramos una coincidencia exacta. ¡Prueba cambiando tu selección!")
    else:
        st.success("¡Aquí están tus cafés recomendados!")
        for _, row in recomendados.iterrows():
            st.markdown(f"### ☕ {row['Nombre']}")
            st.markdown(f"- **Origen:** {row['Origen']}")
            st.markdown(f"- **Proceso:** {row['Proceso']}")
            st.markdown(f"- **Notas:** {row['Notas Sensoriales']}")
            etica = row['Práctica Ética'] if 'Práctica Ética' in row else "No disponible"
            st.markdown(f"- **Intensidad:** {row['Intensidad']} | **Ética:** {etica}")
            if 'Link Shopify' in row and pd.notna(row['Link Shopify']):
                st.markdown(f"[Ver producto en la tienda]({row['Link Shopify']})")
            st.markdown("---")
