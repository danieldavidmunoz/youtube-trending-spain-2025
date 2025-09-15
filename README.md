# TFM - Análisis y Predicción de Videos Trending en YouTube España  

Este repositorio contiene el trabajo de mi **Trabajo de Fin de Máster (TFM)** en Análisis de Datos.  
El objetivo principal es **identificar patrones en los videos trending (relevantes) de YouTube España** y **construir un modelo de Machine Learning** que prediga la probabilidad de que un video se convierta en viral.  

---

## Objetivos

### Objetivos principales
1. **Identificar y explorar los videos trending (relevantes)** en YouTube España a partir de los vídeos obtenidos en la API de la plataforma (categoría `mostPopular`), con una selección de **300 videos diarios**.  
2. **Generar un modelo predictivo** que estime, en términos probabilísticos, las posibilidades de que un vídeo se convierta en trending (relevante).  

 *Nota*: La categoría **trending (relevante)** se define como una métrica meta que pondera principalmente la interacción (likes + comments), sin dejar de lado las visualizaciones.

---

### Objetivos secundarios
1. Crear una **propuesta de indicador de video trending /Índice de Alcance-Engagement/**, tomando los vídeos que cumplen la condición de estar en el **cuarto cuartil** tanto en visionados como en likes y comments.  
2. Explorar e identificar las **correlaciones** de los vídeos que más engagement, comentarios, likes y views generan en relación con:  
   - categorías temáticas,  
   - duración del vídeo,  
   - etiquetas,  
   - y otros factores.  
3. Identificar y detectar **correlaciones significativas en los canales** en relación con número de publicaciones, seguidores y views totales.  
4. **Identificar comunidades de contenido nicho**, agrupando videos o canales según tags, categorías y métricas de engagement mediante modelos de clustering de *machine learning*.  

---

## Estructura del repositorio
- `data/` → datasets en bruto y procesados.  
- `notebooks/` → cuadernos Jupyter con la exploración y el modelado.  
- `src/` → scripts de Python para limpieza, extracción y modelado.  
- `docs/` → documentación y resultados intermedios.  
- `README.md` → este archivo.  

---

## Tecnologías utilizadas
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn)
- **YouTube API v3**
- **Jupyter Notebooks**
- **Git/GitHub**

---

## Cómo usar este repositorio
1. Clonar el repo:  
   ```bash
   git clone https://github.com/danieldavidmunoz/youtube-trending-spain-2025.git
