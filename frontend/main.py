import streamlit as st
import pandas as pd
import requests
from datetime import datetime

API_URL = "http://app:8000/api/v1/list_files"

st.set_page_config(page_title="Listado de archivos", layout="wide")

@st.cache_data(ttl=60)
def fetch_files():
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("objetos", [])
    except Exception as e:
        st.session_state.setdefault("_fetch_error", str(e))
        return []

def human_size(n):
    try:
        n = int(n)
    except:
        return ""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

st.title("Listado de archivos")

with st.sidebar:
    st.header("Controles")
    nit_filter = st.text_input("Filtrar por NIT", value="")
    name_filter = st.text_input("Buscar por nombre", value="")
    refresh = st.button("Refrescar")

if refresh:
    # clear cache for next fetch
    fetch_files.clear()

with st.spinner("Cargando archivos..."):
    objetos = fetch_files()

if not objetos:
    err = st.session_state.get("_fetch_error")
    if err:
        st.error(f"Error al obtener datos: {err}")
    else:
        st.info("No se encontraron objetos.")
    st.stop()

df = pd.DataFrame(objetos)
# Normalize columns that might be missing
for c in ["id", "name", "nit", "size", "url_preassigned", "created_at"]:
    if c not in df.columns:
        df[c] = None

# formatted columns
df["size_human"] = df["size"].apply(human_size)
# parse created_at to nicer format if possible
def fmt_date(s):
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return s
df["created_fmt"] = df["created_at"].apply(fmt_date)

# Apply filters
df_filtered = df.copy()
if nit_filter:
    df_filtered = df_filtered[df_filtered["nit"].astype(str).str.contains(nit_filter, case=False, na=False)]
if name_filter:
    df_filtered = df_filtered[df_filtered["name"].astype(str).str.contains(name_filter, case=False, na=False)]

st.markdown(f"**Total archivos mostrados:** {len(df_filtered)}")

# Table view (reemplazado para tener botones por fila)
table_cols = ["id", "name", "nit", "size_human", "created_fmt"]
col_labels = {
    "id": "ID",
    "name": "Nombre",
    "nit": "NIT",
    "size_human": "Tamaño",
    "created_fmt": "Creado"
}

# Ensure session state for selection
if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = None

# Render a simple table-like layout with a "Ver" button per row
col_widths = [1, 4, 2, 2, 3, 1]  # last column for the button
# Header
hdr_cols = st.columns(col_widths)
for i, key in enumerate(table_cols + ["_action"]):
    label = col_labels.get(key, "") if key != "_action" else ""
    hdr_cols[i].markdown(f"**{label}**")

# Rows
for _, r in df_filtered[table_cols].iterrows():
    row_cols = st.columns(col_widths)
    row_cols[0].write(r["id"])
    row_cols[1].write(r["name"])
    row_cols[2].write(r["nit"])
    row_cols[3].write(r["size_human"])
    row_cols[4].write(r["created_fmt"])
    id_key = f"view_{str(r['id'])}"
    if row_cols[5].button("Ver", key=id_key):
        st.session_state["selected_id"] = r["id"]

# Mostrar detalles si hay selección
if st.session_state.get("selected_id") is None:
    st.info("Haz clic en 'Ver' en alguna fila para ver detalles.")
else:
    selected_id = st.session_state["selected_id"]
    # buscar la fila correspondiente (por si el tipo difiere)
    sel_rows = df_filtered[df_filtered["id"].astype(str) == str(selected_id)]
    if sel_rows.empty:
        st.error("No se encontró el archivo seleccionado.")
    else:
        row = sel_rows.iloc[0]
        st.subheader("Detalles del archivo")
        st.write("Nombre:", row["name"])
        st.write("NIT:", row["nit"])
        st.write("Tamaño:", row["size_human"])
        st.write("Creado:", row["created_fmt"])

        url = row.get("url_preassigned", "")
        if url:
            st.markdown(f"[Abrir en nueva pestaña]({url})", unsafe_allow_html=True)
            st.markdown(f'<a href="{url}" target="_blank" rel="noopener noreferrer"><button>Descargar / Abrir</button></a>', unsafe_allow_html=True)

        with st.expander("Ver JSON completo"):
            st.json(row.dropna().to_dict())