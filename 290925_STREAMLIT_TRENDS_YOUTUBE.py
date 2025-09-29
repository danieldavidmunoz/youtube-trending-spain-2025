# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns        
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import RobustScaler

import textwrap




#st.set_page_config(page_title="Views vs Engagement (Cuadrantes)", layout="centered")

# ---------- Carga de datos ----------
@st.cache_data(show_spinner=False)


def load_df():
    df = pd.read_parquet(
        "/Users/danielmunoz/Documents/EDUCACION/DATA_ANALIST/CURSOS/TFM/DATA/PROCESSED/df_YouTube_2025TFM_2.0.parquet"
    )
    return df

st.set_page_config(page_title="Views vs Engagement (Cuadrantes)", layout="centered")

st.title("📊 Análisis de canales y videos de YouTube")



#--- Views por engagement ---

st.header("Gráfica de Vídeos")

df_1 = load_df().copy()
variables = ["views", "likes", "comments"]
for var in variables:
    fig01=plt.figure(figsize=(8,5))
    sns.histplot(data=df_1, x=var, hue="mostpopular", bins=50, log_scale=True, 
                 palette="coolwarm", alpha=0.6)
    plt.title(f"Distribución de {var}: mostPopular vs. no-mostPopular")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")
    plt.legend(title="mostPopular", labels=["No", "Sí"])
    st.subheader(f"Distribución de {var}: mostPopular vs. no-mostPopular") 
    st.pyplot(fig01) 
    

st.subheader("Distribución de la tasa de engagement con división percentil 95") 



p95 = df_1["engagement_rate"].quantile(0.95)

fig0=plt.figure(figsize=(8,6))
plt.hist(df_1["engagement_rate"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Engagement rate (<= p95)")
plt.axvline(p95, color="red", linestyle="--", linewidth=2, label=f"P90 = {p95:.3f}")
plt.ylabel("Frecuencia")
plt.title("Distribución de la tasa de engagement (sin outliers)")
st.pyplot(fig0) 

#------------------------

st.subheader("Duracion y engagement rate") 


df_1 = load_df().copy()

views_p25, views_p75 = df_1["views"].quantile([0.25, 0.75])
eng_p25, eng_p75     = df_1["engagement_rate"].quantile([0.25, 0.75])

views_p5, views_p95  = df_1["views"].quantile([0.05, 0.95])
eng_p5, eng_p95      = df_1["engagement_rate"].quantile([0.05, 0.95])

# ==========================
# 2. Función de clasificación
# ==========================
def categorize(row):
    v, e = row["views"], row["engagement_rate"]

    # Extremos 5%
    if v <= views_p5 and e <= eng_p5:
        return "LowLow (≤5%)"
    if v >= views_p95 and e >= eng_p95:
        return "HighHigh (≥95%)"

    # Cuadrantes amplios (25/75)
    if v >= views_p75 and e >= eng_p75:
        return "High Views & High Engagement"
    elif v >= views_p75 and e <= eng_p25:
        return "High Views & Low Engagement"
    elif v <= views_p25 and e >= eng_p75:
        return "Low Views & High Engagement"
    elif v <= views_p25 and e <= eng_p25:
        return "Low Views & Low Engagement"

    # En el medio
    return "Middle"

# ==========================
# 3. Aplicar al DataFrame
# ==========================
df_1["quadrant"] = df_1.apply(categorize, axis=1)

fig = plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_1, x="views", y="engagement_rate",
    hue="quadrant", alpha=0.6
)
plt.xscale("log")
plt.title("Videos clasificados por visitas y engagement")
#plt.show()
st.pyplot(fig) 

#--- Crecimiento views por día ---

st.subheader("Outliers por debajo y encima de la dispersion normal") 


X = np.log1p(df_1["views"])   # log(views+1) para estabilizar escala
y = df_1["engagement_rate"]

X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()

df_1["resid"] = ols_model.resid

# ===========================
# 4. Detectar outliers
# ===========================
top_pos = df_1.nlargest(10, "resid")[["title", "views", "engagement_rate", "resid"]]
top_neg = df_1.nsmallest(10, "resid")[["title", "views", "engagement_rate", "resid"]]
fig2=plt.figure(figsize=(9,6))
plt.scatter(df_1["views"], df_1["engagement_rate"], alpha=0.3, label="Otros")
plt.scatter(top_pos["views"], top_pos["engagement_rate"], color="green", label="Outliers + (alto engagement)")
plt.scatter(top_neg["views"], top_neg["engagement_rate"], color="red", label="Outliers - (bajo engagement)")

# Curva LOWESS para referencia
lowess_curve = lowess(y, np.log1p(df_1["views"]), frac=0.3)
plt.plot(np.expm1(lowess_curve[:,0]), lowess_curve[:,1], color="black", linewidth=2)

plt.xscale("log")
plt.xlabel("Views (log)")
plt.ylabel("Engagement rate")
plt.title("Detección de outliers en la relación Views vs Engagement")
plt.legend()
plt.show()
st.pyplot(fig2)


#--- Views y engagement según día de la semana ---

st.subheader("Views y engagement según día de la semana") 

df_1["engagement_rate"] = (df_1["likes"] + df_1["comments"]) / df_1["views"].replace(0, np.nan)

df_1["published_at"] = pd.to_datetime(df_1["published_at"], utc=True)
#df_1["hour"] = df_1["published_at"].dt.hour
df_1["dayofweek"] = df_1["published_at"].dt.day_name()  # 'Monday', 'Tuesday', etc.
df_1["month"] = df_1["published_at"].dt.month
# promedio por hora
# views_by_hour = df_1.groupby("hour")["views"].mean()
# eng_by_hour   = df_1.groupby("hour")["engagement_rate"].mean()

# promedio por día de semana
views_by_day = df_1.groupby("dayofweek")["views"].mean()
eng_by_day   = df_1.groupby("dayofweek")["engagement_rate"].mean()

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Convertir columna a categórica ordenada
df_1["dayofweek"] = pd.Categorical(df_1["published_at"].dt.day_name(),
                                   categories=day_order,
                                   ordered=True)

# Recalcular agregados respetando el orden
views_by_day = df_1.groupby("dayofweek")["views"].mean()
eng_by_day   = df_1.groupby("dayofweek")["comments"].mean()


# === Visualización ===
fig3, ax1 = plt.subplots(figsize=(10,5))

ax1.bar(views_by_day.index, views_by_day.values, alpha=0.6, label="Views promedio")
ax1.set_ylabel("Views (media)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(eng_by_day.index, eng_by_day.values, marker="o", color="red", label="Likes, promedio")
ax2.set_ylabel("Engagement rate (media)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Media de views y engagement según día de la semana")
plt.xticks(rotation=45)
st.pyplot(fig3)

#--- Engagement por categoría ---

st.subheader("Comentarios y likes por 1000 views por categoría ") 

df_1["likes_per_1000views"] = (df_1["likes"] / df_1["views"].replace(0, np.nan)) * 1000
df_1["comments_per_1000views"] = (df_1["comments"] / df_1["views"].replace(0, np.nan)) * 1000

# 2. Agrupar por categoría
stats_by_cat = (
    df_1.groupby("category_id")[["likes_per_1000views", "comments_per_1000views"]]
    .mean()
    .sort_values("likes_per_1000views", ascending=False)
)
category_map = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    31: "Anime/Animation",
    32: "Action/Adventure",
    33: "Classics",
    34: "Comedy",
    35: "Documentary",
    36: "Drama",
    37: "Family",
    38: "Foreign",
    39: "Horror",
    40: "Sci-Fi/Fantasy",
    41: "Thriller",
    42: "Shorts",
    43: "Shows",
    44: "Trailers"
}
# 3. Mapear IDs a nombres
stats_by_cat.index = stats_by_cat.index.map(lambda x: category_map.get(int(x), f"Unknown {x}"))

# 4. Visualización con barras (verticales) + línea
fig4, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(
    stats_by_cat.index,
    stats_by_cat["likes_per_1000views"],
    color="blue", alpha=0.6, label="Likes"
)
ax1.set_ylabel("Likes por cada 1000 views (media)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Línea (comments) en eje secundario
ax2 = ax1.twinx()
ax2.plot(
    stats_by_cat.index,
    stats_by_cat["comments_per_1000views"],
    "ro-", label="Comments"
)
ax2.set_ylabel("Comentarios por cada 1000 views (media)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# --- ROTACIÓN CORRECTA DEL EJE X (sobre ax1) ---
# (opcional) envolver nombres muy largos
wrapped_labels = [textwrap.fill(str(lbl), width=18) for lbl in stats_by_cat.index]
ax1.set_xticks(range(len(wrapped_labels)))
ax1.set_xticklabels(wrapped_labels)
plt.setp(ax1.get_xticklabels(), rotation=90, ha="right")

plt.title("Likes y Comentarios por 1000 views según categoría de vídeo")

# Ajustes para que no se corten etiquetas
fig4.subplots_adjust(bottom=0.35)
# En Streamlit, mejor:
st.pyplot(fig4)


#--- DISTRIBUCION Y OUTLIERS ---

st.subheader("Curva de Lorenz de views") 

views = pd.to_numeric(df_1["views"], errors="coerce").fillna(0).values
n = len(views)
if n == 0:
    raise ValueError("No hay datos de 'views'.")

# --- 1) Ordenar de menor a mayor y calcular acumulados ---
views_sorted = np.sort(views)                    # ascendente
cum_views = np.cumsum(views_sorted)              # acumulado de views
total_views = cum_views[-1] if cum_views[-1] > 0 else 1.0

# Eje X: proporción acumulada de vídeos (0..1)
cum_videos = np.arange(1, n+1) / n
# Eje Y: proporción acumulada de views (0..1)
lorenz = cum_views / total_views
lorenz = np.insert(lorenz, 0, 0)                 # forzar origen (0,0)
cum_videos0 = np.insert(cum_videos, 0, 0)

# --- 2) Gini (área entre igualdad y Lorenz): G = 1 - 2 * AUC(Lorenz) ---
auc_lorenz = np.trapz(lorenz, cum_videos0)
gini = 1 - 2 * auc_lorenz

# --- 3) ¿Qué % de vídeos concentra el 80% de las views? ---
threshold = 0.80
# índice del primer punto donde el acumulado de views >= 80%
idx_80 = np.searchsorted(lorenz, threshold)
prop_videos_80 = cum_videos0[idx_80] * 100   # en porcentaje

print(f"Gini de views: {gini:.3f}")
print(f"Proporción de vídeos necesaria para acumular el 80% de views: {prop_videos_80:.1f}%")

# --- 4) Gráfica de Lorenz ---
fig5=plt.figure(figsize=(7,6))
plt.plot(cum_videos0, lorenz, label="Curva de Lorenz (views)")
plt.plot([0,1], [0,1], linestyle="--", label="Igualdad perfecta")
# marcar el punto del 80%
plt.axhline(threshold, color="gray", linestyle=":")
plt.axvline(cum_videos0[idx_80], color="gray", linestyle=":")
plt.scatter([cum_videos0[idx_80]], [threshold], color="red", zorder=3,
            label=f"{prop_videos_80:.1f}% vídeos → 80% views")

plt.xlabel("Proporción acumulada de vídeos")
plt.ylabel("Proporción acumulada de views")
plt.title(f"Curva de Lorenz de views (Gini = {gini:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
st.pyplot(fig5)

#--- DURACION Y ENGAGEMENT ---

st.subheader("Duración y Engagement") 

fig6=plt.figure(figsize=(8,5))
sns.scatterplot(data=df_1, x="duration_minutes", y="engagement_rate", alpha=0.4)
plt.xlabel("Duración del vídeo (minutos)")
plt.ylabel("Engagement (likes+comments / views)")
plt.title("Relación entre duración y engagement")
#plt.ylim(0, df_1["engagement_rate"].quantile(0.95))  # cortar outliers extremos
st.pyplot(fig6)

st.subheader("Dispersion en el engagement entre videos cortos y de mas de un minuto") 

df_1["duracion_tipo"] = pd.cut(
    df_1["duration_minutes"],
    bins=[0, 1, df_1["duration_minutes"].max()],
    labels=["Muy cortos (<=1 min)", "Más de 1 min"]
)

fig7=plt.figure(figsize=(6,5))
sns.boxplot(data=df_1, x="duracion_tipo", y="engagement_rate")
plt.xlabel("Tipo de vídeo")
plt.ylabel("engagement_rate")
plt.title("Comparación de engagement en vídeos cortos vs largos")
plt.ylim(0, df_1["engagement_rate"].quantile(0.95))  # quitar outliers extremos
plt.show()
st.pyplot(fig7)


#--- VIEWS SEGUN CAPTION Y ENGAGEMENT ---

st.subheader("Dispersion en los views si tienen subtitulos o no") 

fig8=plt.figure(figsize=(6,5))
sns.boxplot(data=df_1, x="has_caption", y="views")
plt.ylim(0, df_1["views"].quantile(0.95))  # opcional: limitar outliers
plt.xlabel("¿Tiene captions?")
plt.ylabel("Views")
plt.title("Comparación de visitas según presencia de captions")
plt.show()
st.pyplot(fig8)

#--- ENGAGEMENT POR CATEGORIAS ---
st.subheader("Correlación entre features") 
df_EDA_numericas_bruto=df_1[['duration_minutes','views', 'likes' , 'comments' , 'tags_count',  
                           'subscriber_count','channel_video_count', 'channel_views', 
                           'channel_age_days']]
corr_matrix = df_EDA_numericas_bruto.corr()
fig08=plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.title("Correlación entre features")
st.pyplot(fig08)
#--- ENGAGEMENT POR CATEGORIAS ---

st.subheader("Engagement total por categorias") 
df_1["category_name"] = df_1["category_id"].map(category_map)

engagement_total = (
    df_1.groupby("category_name")["engagement_rate"]
        .sum()
        .reset_index()
        .sort_values("engagement_rate", ascending=False)
)

# Pie chart
fig9=plt.figure(figsize=(15,15))
plt.pie(
    engagement_total["engagement_rate"],
    labels=engagement_total["category_name"],
    autopct="%1.1f%%",
    startangle=180
)
plt.title("Proporción del engagement total por categoría")

st.pyplot(fig9)


st.subheader("engagement medio por categoría") 

df_1["category_name"] = df_1["category_id"].map(category_map)

# --- 2) Calcular engagement medio por categoría ---
ranking = (
    df_1.groupby("category_name")["engagement_rate"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
)

ranking_2 = (
    df_1.groupby("category_name")
    .agg(num_videos=("video_id", "count"))   # nº de vídeos por categoría
    .sort_values("num_videos", ascending=False)  # ordenar de mayor a menor
    .reset_index()
)

st.subheader("Total de videos por categorías") 
st.table(ranking_2)


st.subheader("Engagement rate por categorías") 
fig10=plt.figure(figsize=(10,6))
sns.barplot(data=ranking, x="engagement_rate", y="category_name", palette="viridis")
plt.xlabel("Engagement medio (likes+comments / views)")
plt.ylabel("Categoría")
plt.title("Ranking de categorías por engagement")
plt.show()
st.pyplot(fig10)


st.subheader("Comparacion de medianas de likes por categorías") 

ranking_medianas = (
    df_1.groupby("category_name")[["likes_per_view", "comments_per_view"]]
        .median()
        .reset_index()
)

# --- Calcular medias ---
ranking_medias = (
    df_1.groupby("category_name")[["likes_per_view", "comments_per_view"]]
        .mean()
        .reset_index()
)

# --- Ordenar por likes_per_view (para que ambos tengan mismo orden) ---
order = ranking_medianas.sort_values("likes_per_view", ascending=False)["category_name"]
ranking_medianas = ranking_medianas.set_index("category_name").loc[order].reset_index()
ranking_medias = ranking_medias.set_index("category_name").loc[order].reset_index()

# --- Ajustar escalas iguales ---
ymax_likes = max(ranking_medianas["likes_per_view"].max(), ranking_medias["likes_per_view"].max())
ymax_comments = max(ranking_medianas["comments_per_view"].max(), ranking_medias["comments_per_view"].max())

x = np.arange(len(order))

fig11, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)

# --- Gráfico 1: Median ---
ax1 = axes[0]
ax1.bar(x - 0.2, ranking_medianas["likes_per_view"], width=0.4, color="tab:blue", label="Likes/View")
ax1.set_ylabel("Mediana Likes/View", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, ymax_likes*1.1)

ax2 = ax1.twinx()
ax2.bar(x + 0.2, ranking_medianas["comments_per_view"], width=0.4, color="tab:orange", label="Comments/View")
ax2.set_ylabel("Mediana Comments/View", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(0, ymax_comments*1.1)

ax1.set_title("Comparación de medianas Likes/View y Comments/View por categoría")

# --- Gráfico 2: Media ---
ax3 = axes[1]
ax3.bar(x - 0.2, ranking_medias["likes_per_view"], width=0.4, color="tab:blue", label="Likes/View")
ax3.set_ylabel("Media Likes/View", color="tab:blue")
ax3.tick_params(axis="y", labelcolor="tab:blue")
ax3.set_ylim(0, ymax_likes*1.1)

ax4 = ax3.twinx()
ax4.bar(x + 0.2, ranking_medias["comments_per_view"], width=0.4, color="tab:orange", label="Comments/View")
ax4.set_ylabel("Media Comments/View", color="tab:orange")
ax4.tick_params(axis="y", labelcolor="tab:orange")
ax4.set_ylim(0, ymax_comments*1.1)

ax3.set_title("Comparación de medias Likes/View y Comments/View por categoría")

# Eje X común
ax3.set_xticks(x)
ax3.set_xticklabels(order, rotation=45, ha="right")

plt.tight_layout()
plt.show()

st.pyplot(fig11)


#-----------------ENGAGEMENT RAROS



st.subheader("Alto engagement y poco view de la categoría mostpopular") 


base = df_1[df_1['mostpopular'] == 1]
base = df_1[df_1['mostpopular'].fillna(False).astype(bool)]
low_views_threshold = df_1[df_1["mostpopular"]==1]["views"].quantile(0.25)
candidatos = df_1[(df_1["views"] <= low_views_threshold) & 
                  (df_1["engagement_rate"] >= df_1["engagement_rate"].quantile(0.95)) & (df_1['mostpopular']== True)]

fig12=plt.figure(figsize=(7,5))
sns.scatterplot(data=base, x="views", y="engagement_rate", alpha=0.5)

sns.scatterplot(
    data=candidatos,
    x="views", y="engagement_rate",
    color="red", label="Alt. engagement + Pocos views en MostPopular"
)


#plt.xscale("log")  # opcional
plt.xlabel("Views")
plt.ylabel("Engagement rate")
plt.title("MostPopular: alto engagement con pocos views")
plt.legend()
plt.show()
st.pyplot(fig12)


st.subheader("Videos de <50000 views con alto engagement") 


candidatos_2=df_1[(
    df_1['views']<=50000) & (
        df_1['engagement_rate'] >= df_1['engagement_rate'].quantile(0.95)
    )
    ]
fig13=plt.figure(figsize=(7,5))
sns.scatterplot(data=df_1, x="views", y="engagement_rate", alpha=0.5)

sns.scatterplot(
    data=candidatos_2,
    x="views", y="engagement_rate",
    color="red", label="Alt. engagement + Pocos views"
)

plt.xscale("log")  # opcional
plt.xlabel("Views")
plt.ylabel("Engagement rate")
plt.title("Alto engagement con menos de 50000 views")
plt.legend()
plt.show()
st.pyplot(fig13)


#=======================IMPACTo Y RANKEADO================

st.subheader("Relacion entre impacto y engagement") 


df_1["Impacto"] = df_1["engagement_rate"] * np.log1p(df_1["views"])

fig14=plt.figure(figsize=(10,7))
scatter = plt.scatter(
    x=df_1["views"],
    y=df_1["engagement_rate"],
    s=df_1["Impacto"]*200,    # tamaño de la burbuja (ajusta factor si queda muy grande/pequeño)
    c=df_1["Impacto"],        # color = impacto
    cmap="viridis",
    alpha=0.6,
    edgecolor="k"
)

plt.xscale("log")  # log en views para comprimir rango
plt.xlabel("Views (escala log)")
plt.ylabel("Engagement rate")
plt.title("Mapa de Impacto de vídeos en YouTube")

cbar = plt.colorbar(scatter)
cbar.set_label("Impacto (engagement_rate × log(views))")

st.pyplot(fig14)

#============================================

st.subheader("20 video con mas impacto y engagement") 


top20 = df_1.sort_values("Impacto", ascending=False).head(20)

# Mostrar en tabla resumida
print(top20[["title", "category_name", "views", "likes", "comments", "engagement_rate", "Impacto"]])

# --- 3) Visualización ---
fig15=plt.figure(figsize=(10,6))
sns.barplot(data=top20, x="Impacto", y="title", hue="category_name", dodge=False, palette="Paired")
plt.xlabel("Índice de Impacto")
plt.ylabel("Vídeo")
plt.title("Top 20 vídeos por Impacto")
plt.legend(title="Categoría", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
st.pyplot(fig15)



#=========================================


st.subheader("Impacto según mediana por categoría") 
impacto_categoria_media = (
    df_1.groupby("category_name")["Impacto"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
)
impacto_stats = (
    df_1.groupby("category_name")["Impacto"]
        .agg(Impacto_mediana="median", Impacto_total="sum", n="size")
        .reset_index()
        .sort_values("Impacto_total", ascending=False))

fig16=plt.figure(figsize=(10,6))
sns.barplot(data=impacto_categoria_media, x="Impacto", y="category_name", palette="viridis")
plt.xlabel("Impacto medio por vídeo")
plt.ylabel("Categoría")
plt.title("Ranking de categorías por Impacto medio")
st.pyplot(fig16)


#======================================

st.subheader("Impacto total por categoría y según mediana") 

x = np.arange(len(impacto_stats["category_name"]))



fig17, ax1 = plt.subplots(figsize=(14,6))

ax1.bar(x, impacto_stats["Impacto_total"], color="tab:blue", alpha=0.6)
ax1.set_ylabel("Impacto total", color="tab:blue")

# Línea roja: impacto medio
ax2 = ax1.twinx()
ax2.plot(x, impacto_stats["Impacto_mediana"], color="tab:red", marker="o")
ax2.set_ylabel("Impacto mediana", color="tab:red")

# 👇 aquí fuerzas las etiquetas con rotación vertical
ax1.set_xticks(x)
ax1.set_xticklabels(impacto_stats["category_name"], rotation=90, ha="center")

plt.title("Impacto total y mediana por categoría")
plt.tight_layout()
plt.show()

st.pyplot(fig17)

#============================================================
#======================== CANALES ===========================
st.header("Gráfica de Canales")

st.subheader("Top 20 vídeos más vistos (uno por canal)") 


df_top_por_canal_views = (df_1
    .sort_values(["channel_id","views","likes","published_at"], ascending=[True, False, False, False])
    .drop_duplicates("channel_id", keep="first"))
df_top_por_canal_views["video_label"] = df_top_por_canal_views["title"] + " | " + df_top_por_canal_views["channel_title"]
top_videos = df_top_por_canal_views.sort_values("views", ascending=False).head(20)

fig18=plt.figure(figsize=(12,8))
sns.barplot(
    data=top_videos,
    x="views",
    y="video_label",   # 👈 título + canal
    palette="viridis"
)
plt.xlabel("Views")
plt.ylabel("Vídeo | Canal")
plt.title("Top 20 vídeos más vistos (uno por canal)")
plt.tight_layout()
st.pyplot(fig18)


#------------------------------------
st.subheader("Medición a partir del índice compuesto de éxito del canal")
scaler = RobustScaler()
canal_stats = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(efficiency=("efficiency","mean"),
         engagement_subscribers=("engagement_subscribers","mean"))
)

canal_stats[["efficiency","engagement_subscribers"]] = (
    canal_stats[["efficiency","engagement_subscribers"]].fillna(0.0)
)
canal_stats[["eff_norm","engsub_norm"]] = scaler.fit_transform(
    canal_stats[["efficiency","engagement_subscribers"]]
)
# índice ponderado (70% eficiencia, 30% engagement_subscribers)
canal_stats["indice"] = 0.5 * canal_stats["eff_norm"] + 0.5 * canal_stats["engsub_norm"]
ranking_3 = canal_stats.sort_values("indice", ascending=False).head(20)

st.table(ranking_3)
#----------------------------

st.subheader("Top de 20 canales segun puntuacion en índice compuesto por éxito del canal")
topN = 20
ranking = canal_stats.sort_values("indice", ascending=False).head(topN)

fig19=plt.figure(figsize=(12,8))
sns.barplot(
    data=ranking,
    x="indice",
    y="channel_title",
    palette="viridis"
)
plt.xlabel("Índice de Eficiencia (ajustado)")
plt.ylabel("Canal")
plt.title(f"Top {topN} canales por índice de eficiencia")
plt.tight_layout()
plt.show()
st.pyplot(fig19)

#----------------------------ENGAGEMENT RELATIVO

st.subheader("Engagement relativo (interacciones / suscriptores)")
ranking_engsub = df_1.sort_values("engagement_subscribers", ascending=False).head(20)
fig20=plt.figure(figsize=(12,8))
sns.barplot(
    data=ranking_engsub,
    x="engagement_subscribers",
    y="channel_title",
    palette="mako"
)
plt.xlabel("Engagement relativo (interacciones / suscriptores)")
plt.ylabel("Canal")
plt.title("Top 20 canales con mayor engagement relativo")
plt.tight_layout()
plt.show()
st.pyplot(fig20)

#----------------------------ENGAGEMENT RELATIVO
st.subheader("Engagement relativo vs Eficiencia de canales")
          

canal_stats = (df_1.groupby(["channel_id","channel_title"], as_index=False)
    .agg(efficiency=("efficiency","mean"),
         engagement_subscribers=("engagement_subscribers","mean"))
)

fig21=plt.figure(figsize=(10,7))
sns.scatterplot(
    data=canal_stats,
    x="efficiency", 
    y="engagement_subscribers",
#    size="indice_compuesto",       # si ya calculaste índice combinado
    hue="engagement_subscribers",
    sizes=(50, 400),
    palette="viridis",
    alpha=0.7
)
plt.xlabel("Efficiency (likes+comments / nº vídeos)")
plt.ylabel("Engagement relativo (likes+comments / suscriptores)")
plt.title("Engagement relativo vs Eficiencia de canales")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()
st.pyplot(fig21)

#----------------------------ENGAGEMENT RELATIVO

st.subheader("Canales pequeños con seguidores fieles")
canal_stats = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(efficiency=("efficiency","mean"),
         engagement_subscribers=("engagement_subscribers","mean"))
)

# 2) Traer el número de suscriptores por canal (máximo observado)
subs_por_canal = (df_1
    .groupby(["channel_id"], as_index=False)
    .agg(subscribers=("subscriber_count","max"))
)

# 3) Merge -> ahora canal_stats tiene 'subscribers'
canal_stats = canal_stats.merge(subs_por_canal, on="channel_id", how="left")

# 4) Definir umbrales para "canal pequeño" y "fiel"
low_subs = canal_stats["subscribers"].quantile(0.25)           # 25% inferior en subs
high_engsub = canal_stats["engagement_subscribers"].quantile(0.75)  # 25% superior en engagement relativo

# 5) Filtrar canales pequeños con comunidad fiel
canales_fieles = canal_stats[
    (canal_stats["subscribers"] <= low_subs) &
    (canal_stats["engagement_subscribers"] >= high_engsub)
].copy()

# 6) Scatter general + destacados
fig22=plt.figure(figsize=(10,7))
ax = sns.scatterplot(
    data=canal_stats,
    x="subscribers",
    y="engagement_subscribers",
    alpha=0.45
)

# resaltar en rojo los pequeños pero fieles
sns.scatterplot(
    data=canales_fieles,
    x="subscribers",
    y="engagement_subscribers",
    s=120, color="red", edgecolor="k", label="Pequeños y fieles", ax=ax
)

plt.xscale("log")  # subs suele estar muy sesgado
plt.xlabel("Suscriptores (escala log)")
plt.ylabel("Engagement relativo (likes+comments / suscriptores)")
plt.title("¿Hay canales pequeños con seguidores muy fieles?")
plt.legend()
plt.tight_layout()
plt.show()
st.pyplot(fig22)


#----------------------------COMPARATIVA CANALES
st.subheader("Top 20 canales con más visitas promedio por vídeo publicado")


canal_views = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(channel_views=("channel_views","max"),
         channel_video_count=("channel_video_count","max"))
)

# Evitar divisiones por 0
canal_views["views_per_video"] = np.where(
    canal_views["channel_video_count"] > 0,
    canal_views["channel_views"] / canal_views["channel_video_count"],
    0
)

ranking = canal_views.sort_values("views_per_video", ascending=False)
#print(ranking[["channel_title","channel_views","channel_video_count","views_per_video"]].head(10))
topN = 20
fig23=plt.figure(figsize=(12,8))
sns.barplot(
    data=ranking.head(topN),
    x="views_per_video",
    y="channel_title",
    palette="viridis"
)
plt.xlabel("Visitas promedio por vídeo")
plt.ylabel("Canal")
plt.title(f"Top {topN} canales con más visitas promedio por vídeo publicado")
plt.tight_layout()
plt.show()
st.pyplot(fig23)

#----------------------------Frecuencia de publicación

st.subheader("Distribución de la frecuencia de publicación todos los canales")
canal_stats = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(channel_video_count=("channel_video_count","max"),
         channel_age_days=("channel_age_days","max"),
         efficiency=("efficiency","mean"),
         engagement_subscribers=("engagement_subscribers","mean"))
)
# Frecuencia de publicación = vídeos subidos por día
canal_stats["frecuencia"] = np.where(
    canal_stats["channel_age_days"] > 0,
    canal_stats["channel_video_count"] / canal_stats["channel_age_days"],
    0
)

fig24=plt.figure(figsize=(10,6))
sns.histplot(
    data=canal_stats,
    x="frecuencia",
    bins=30,
    log_scale=(True, False),   # log en X
    color="skyblue",
    edgecolor="black"
)

plt.xlabel("Frecuencia de publicación (vídeos/día, escala log)")
plt.ylabel("Número de canales")
plt.title("Distribución de la frecuencia de publicación (todos los canales)")
plt.tight_layout()
plt.show()
st.pyplot(fig24)



#----------------------------Dispersión Frecuencia de publicación

st.subheader("Dispersión de la frecuencia de publicación de los 50 canales que mas publican")
top_activos = (canal_stats
    .sort_values(["frecuencia","engagement_subscribers"], ascending=[False,False])
    .head(50)
)
print(top_activos[["channel_title","frecuencia","engagement_subscribers"]])
fig25=plt.figure(figsize=(8,4))
sns.boxplot(x=top_activos["frecuencia"], color="lightblue")
#plt.xscale("log")
plt.xlabel("Frecuencia de publicación (vídeos/día)")
plt.title("Distribución de frecuencia de publicación (boxplot)")
plt.tight_layout()
plt.show()
st.pyplot(fig25)



#----------------------------Dispersión Frecuencia de publicación

st.subheader("Distribución de la frecuencia de publicación (sin outliers)")

umbral = canal_stats["frecuencia"].quantile(0.95)
fig26=plt.figure(figsize=(8,5))
sns.histplot(canal_stats[canal_stats["frecuencia"] <= umbral]["frecuencia"], bins=30, color="skyblue")
plt.xlabel("Frecuencia de publicación (vídeos/día)")
plt.ylabel("Número de canales")
plt.title("Distribución de la frecuencia de publicación (sin outliers)")
plt.tight_layout()
plt.show()
st.pyplot(fig26)



#---------------------------- Frecuencia de publicación y visualizaciones  totales

st.subheader("Relación entre frecuencia de publicación y visualizaciones totales")
canal_stats = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(channel_views=("channel_views","max"),
         channel_video_count=("channel_video_count","max"),
         channel_age_days=("channel_age_days","max"))
)

# Frecuencia de publicación = vídeos / días
canal_stats["frecuencia"] = np.where(
    canal_stats["channel_age_days"] > 0,
    canal_stats["channel_video_count"] / canal_stats["channel_age_days"],
    0)

fig27=plt.figure(figsize=(10,7))
sns.scatterplot(
    data=canal_stats,
    x="frecuencia",
    y="channel_views",
    size="channel_video_count",  # tamaño según nº de vídeos
    hue="frecuencia",
    palette="viridis",
    alpha=0.6
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frecuencia de publicación (vídeos/día, log)")
plt.ylabel("Channel Views totales (log)")
plt.title("Relación entre frecuencia de publicación y visualizaciones totales")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()
st.pyplot(fig27)



#---------------------------- Frecuencia de publicación y visualizaciones  totales

st.subheader("Relación entre frecuencia de publicación e impacto")
canal_stats["views_per_video"] = np.where(
    canal_stats["channel_video_count"] > 0,
    canal_stats["channel_views"] / canal_stats["channel_video_count"],
    0
)
freq_low =  canal_stats["frecuencia"].quantile(0.25)
freq_high = canal_stats["frecuencia"].quantile(0.75)
impact_low = canal_stats["views_per_video"].quantile(0.25)
impact_high = canal_stats["views_per_video"].quantile(0.75)

# Filtrar canales de interés
canales_mucho_poco = canal_stats[
    (canal_stats["frecuencia"] >= freq_high) &
    (canal_stats["views_per_video"] <= impact_low)
]
canales_poco_mucho = canal_stats[
    (canal_stats["frecuencia"] <= freq_low) &
    (canal_stats["views_per_video"] >= impact_high)
]

canales_mucho_mucho = canal_stats[
    (canal_stats["frecuencia"] >= freq_high) &
    (canal_stats["views_per_video"] >= impact_high)
]

canales_poco_poco = canal_stats[
    (canal_stats["frecuencia"] <= freq_low) &
    (canal_stats["views_per_video"] <= impact_low)
]
fig28=plt.figure(figsize=(10,7))

# Base: todos los canales en gris claro
sns.scatterplot(
    data=canal_stats,
    x="frecuencia",
    y="views_per_video",
    color="lightgray",
    alpha=0.4,
    label="Otros canales"
)

# 1) Mucho contenido, bajo impacto
sns.scatterplot(
    data=canales_mucho_poco,
    x="frecuencia",
    y="views_per_video",
    color="red",

    label="Mucho contenido, bajo impacto"
)

sns.scatterplot(
    data=canales_poco_poco,
    x="frecuencia",
    y="views_per_video",
    color="purple",

    label="Poco contenido, bajo impacto")

# 2) Poco contenido, mucho impacto
sns.scatterplot(
    data=canales_poco_mucho,
    x="frecuencia",
    y="views_per_video",
    color="blue",
 
    label="Poco contenido, alto impacto"
)

# 3) Mucho contenido, mucho impacto
sns.scatterplot(
    data=canales_mucho_mucho,
    x="frecuencia",
    y="views_per_video",
    color="green",

    label="Mucho contenido, alto impacto"
)

# Escalas logarítmicas (muy útil aquí)
plt.xscale("log")
plt.yscale("log")

# Líneas de referencia para umbrales
plt.axvline(freq_high, color="black", linestyle="--", alpha=0.6)
plt.axhline(impact_high, color="black", linestyle="--", alpha=0.6)

plt.xlabel("Frecuencia de publicación (vídeos/día, log)")
plt.ylabel("Visitas promedio por vídeo (log)")
plt.title("Canales: Frecuencia vs Impacto por vídeo (cuadrantes)")
plt.legend()
plt.tight_layout()
plt.show()

st.pyplot(fig28)



#---------------------------- Tiempo de creación y éxito del canal

st.subheader("Relación entre la edad del canal e influencia en el éxito")

canal_age = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(channel_age_days=("channel_age_days","max"),
         channel_views=("channel_views","max"),
         channel_video_count=("channel_video_count","max")))

fig29=plt.figure(figsize=(10,12))
sns.scatterplot(data=canal_age, x="channel_age_days", y="channel_views")

st.pyplot(fig29)



#---------------------------- Tiempo de creación y éxito del canal

st.subheader("Views acumulados según la antigüedad del canal")
df_bins = canal_age.copy()
df_bins["age_years"] = df_bins["channel_age_days"] / 365
bins = [0,1,3,5,10,20]
labels = ["<1año","1–3a","3–5a","5–10a","10–20a"]
df_bins["age_group"] = pd.cut(df_bins["age_years"], bins=bins, labels=labels, right=False)
trend = (df_bins.groupby("age_group")["channel_views"]
         .median()
         .reset_index())
         
fig30=plt.figure(figsize=(10,6))
sns.violinplot(data=df_bins, x="age_group", y="channel_views", inner="quartile")
plt.yscale("log")
plt.xlabel("Antigüedad del canal (años)")
plt.ylabel("Views acumulados (log)")
plt.title("Distribución de views por antigüedad del canal")
plt.tight_layout()
plt.show()
st.pyplot(fig30)



#---------------------------- Tiempo de creación y éxito del canal

st.subheader("Tendencia de views acumulados por antigüedad del canal")


fig31=plt.figure(figsize=(8,5))
sns.lineplot(data=trend, x="age_group", y="channel_views", marker="o")
plt.yscale("log")
plt.xlabel("Antigüedad del canal (años)")
plt.ylabel("Views medianos acumulados (log)")
plt.title("Tendencia de views acumulados por antigüedad del canal")
plt.tight_layout()
plt.show()
st.pyplot(fig31)



#---------------------------- Tiempo de creación y éxito del canal

st.subheader("Impacto medio de los canales según su antigüedad")
# Impacto medio por canal
canal_impacto = (df_1
    .groupby(["channel_id","channel_title"], as_index=False)
    .agg(channel_age_days=("channel_age_days","max"),
         Impacto_medio=("Impacto","median"))   # uso mediana para reducir outliers
)
# Convertimos a años para agrupar
canal_impacto["age_years"] = canal_impacto["channel_age_days"] / 365

bins = [0,1,3,5,10,20]
labels = ["<1año","1–3a","3–5a","5–10a","10–20a"]
canal_impacto["age_group"] = pd.cut(canal_impacto["age_years"], bins=bins, labels=labels, right=False)

fig32=plt.figure(figsize=(10,6))

# Boxplot de impacto por grupo de edad
sns.boxplot(data=canal_impacto, x="age_group", y="Impacto_medio", color="lightblue")

# Línea con medianas
medianas = canal_impacto.groupby("age_group")["Impacto_medio"].median().reset_index()
sns.lineplot(data=medianas, x="age_group", y="Impacto_medio", marker="o", color="red", label="Mediana Impacto")

plt.yscale("log")   # impacto puede estar muy disperso
plt.xlabel("Antigüedad del canal (años)")
plt.ylabel("Impacto medio por canal (log)")
plt.title("Impacto medio de los canales según su antigüedad")
plt.legend()
plt.tight_layout()
st.pyplot(fig32)



#---------------------------- DISTRIBUCION Y OUTLIERS


canal_impacto_un_video = df_1[df_1["channel_video_count"] > 1].copy()

canal_stats = (
    canal_impacto_un_video
    .groupby(["channel_id", "channel_title"], as_index=False)
    .agg(
        views_mean=("views", "mean"),
        views_max=("views", "max"),
        views_sum=("views", "sum"),
        n_videos=("video_id", "count")
    )
)

# evitar divisiones raras
canal_stats = canal_stats.replace({"views_mean": {0: np.nan}, "views_sum": {0: np.nan}})

canal_stats["max_share"]   = canal_stats["views_max"] / canal_stats["views_sum"]
canal_stats["max_vs_mean"] = canal_stats["views_max"] / canal_stats["views_mean"]

# comprueba que existe
assert "max_vs_mean" in canal_stats.columns, canal_stats.columns.tolist()

# top 20 limpio (quita NaN/inf por si acaso)
top20 = (
    canal_stats
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["max_vs_mean"])
    .nlargest(20, "max_vs_mean")
    .sort_values("max_vs_mean", ascending=False)
)

# ejemplo de gráfico (barh) del top 20
fig = plt.figure(figsize=(8,6))
plt.barh(top20["channel_title"], top20["max_vs_mean"], edgecolor="black")
plt.gca().invert_yaxis()
plt.xlabel("views_max / views_mean")
plt.title("Top 20 canales por max_vs_mean")
st.pyplot(fig)

# histograma de max_share (como antes)
fig33 = plt.figure(figsize=(6,4))
plt.hist(canal_stats["max_share"].dropna(), bins=20, edgecolor="black")
plt.title("Distribución de dependencia en views de un solo vídeo")
plt.xlabel("Proporción views (max_share)")
plt.ylabel("Número de canales")
st.pyplot(fig33)



#---------------------------- DISTRIBUCION Y OUTLIERS

st.subheader("Los 20 canales que más impacto tienen del corpus")

top20_sorted = top20.sort_values("max_vs_mean", ascending=False)

fig34, ax = plt.subplots(figsize=(10, 6))   # crea la figura
ax.bar(top20_sorted["channel_title"], top20_sorted["max_vs_mean"], edgecolor="black")
ax.set_xlabel("Canal")
ax.set_ylabel("views_max / views_mean")
ax.set_title("Top 20 canales por max_vs_mean")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()

st.pyplot(fig34)

#---------------------------- RELACION ENTRE VIEWS Y ENGAGEMENT

st.subheader("Relación entre views y engagement por canal")
df_1["engagement_rate"] = (df_1["likes"] + df_1["comments"]) / df_1["views"].replace(0, 1)
canal_stats2 = (
    df_1.groupby(["channel_id", "channel_title"], as_index=False)
    .agg(
        views_sum=("views", "sum"),
        engagement_mean=("engagement_rate", "mean"),
        n_videos=("video_id", "count")
))
canal_stats2 = canal_stats2[canal_stats2["views_sum"] > 100]  # ejemplo: solo canales con >100 views totales
fig35=plt.figure(figsize=(7,5))
plt.scatter(canal_stats2["views_sum"], canal_stats2["engagement_mean"], alpha=0.5)
plt.xscale("log")  # porque views suele ser muy desigual
plt.xlabel("Total de views del canal (log)")
plt.ylabel("Engagement medio (likes+comments / views)")
plt.title("Relación entre views y engagement por canal")

plt.ylim(0, 0.3)  # mostrar solo hasta 20% de engagement
plt.show()
st.pyplot(fig35)

#---------------------------- RELACION ENTRE VIEWS Y ENGAGEMENT

st.subheader("Total de views por suscriptores ")
canal_stats3 = (
    df_1.groupby(["channel_id", "channel_title"], as_index=False)
    .agg(
        channel_views=("channel_views", "max"),
    
        subscriber_count=("subscriber_count", "max")
))
fig36=plt.figure(figsize=(10, 15))
sns.scatterplot(data=canal_stats3, x="subscriber_count", y="channel_views", alpha=0.5)
plt.xscale("log")
plt.xlabel("Suscriptores Total de views del canal (log)")
plt.yscale("log")
plt.ylabel(" Total de views del canal (log)")
plt.show()

st.pyplot(fig36)

#---------------------------- VIRALIDAD INESPERADA

st.subheader("Distribución del ratio de viralidad por canal")

canal_mean = (
    df_1.groupby("channel_id", as_index=False)
    .agg(media_views_per_day=("views_per_day", "mean"))
)
df_viral = df_1.merge(canal_mean, on="channel_id", how="left")
df_viral["viralidad_ratio"] = df_viral["views_per_day"] / df_viral["media_views_per_day"]
umbral = df_viral["viralidad_ratio"].quantile(0.99)
videos_virales = df_viral[df_viral["viralidad_ratio"] >= umbral]
top_canales = df_viral["channel_id"].value_counts().nlargest(20).index
df_top = df_viral[df_viral["channel_id"].isin(top_canales)]

fig37=plt.figure(figsize=(14,6))
sns.boxplot(data=df_top, x="channel_title", y="viralidad_ratio")
plt.xticks(rotation=90)
plt.yscale("log")  # log para ver mejor los outliers
plt.title("Distribución del ratio de viralidad por canal")
plt.ylabel("Viralidad ratio (views_per_day / media canal)")
plt.xlabel("Canal")
plt.show()
st.pyplot(fig37)

#---------------------------- VIRALIDAD INESPERADA

st.subheader("Scatterplot: Views por día del vídeo vs. media histórica del canal")
lim = max(df_viral["media_views_per_day"].max(), df_viral["views_per_day"].max())
lim = df_viral[["media_views_per_day", "views_per_day"]].max().max()  # límite para la diagonal

fig38, ax = plt.subplots(figsize=(8,6))

# Diagonal de referencia
ax.plot([1, lim], [1, lim], color="red", linestyle="--")

# Scatter
sns.scatterplot(
    data=df_viral, 
    x="media_views_per_day", 
    y="views_per_day", 
    alpha=0.4, 
    ax=ax
)

# Escalas log
ax.set_xscale("log")
ax.set_yscale("log")

# Etiquetas
ax.set_xlabel("Media de views_per_day del canal")
ax.set_ylabel("Views_per_day del vídeo")
ax.set_title("Scatterplot: Views por día del vídeo vs. media histórica del canal")

plt.tight_layout()
st.pyplot(fig38)
#---------------------------- VIRALIDAD INESPERADA

st.subheader("Scatterplot: Views por día del vídeo vs. media histórica del canal")
videos_virales = df_viral[df_viral["viralidad_ratio"] >= umbral]
fig39=plt.figure(figsize=(12,6))
sns.barplot(data=videos_virales, x="channel_title", y="viralidad_ratio", estimator="mean", ci=None)
plt.xticks(rotation=90)
plt.yscale("log")
plt.title("Promedio de viralidad (ratio) por canal en vídeos virales")
plt.show()
st.pyplot(fig39)

fig.write_html("/Users/danielmunoz/Documents/EDUCACION/DATA_ANALIST/CURSOS/TFM/REPORTS/grafico.html")