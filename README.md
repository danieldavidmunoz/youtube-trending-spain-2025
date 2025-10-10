# TFM - Análisis y Predicción de Videos Trending en YouTube España  

Este repositorio contiene el trabajo de mi Trabajo de Fin de Máster (TFM) en Análisis de Datos.
El objetivo principal es identificar patrones en los videos trending (relevantes) de YouTube España y construir un modelo de Machine Learning que prediga la probabilidad de que un video se convierta en viral. 

---

## Objetivos

### Objetivos general

Desarrollar un modelo de análisis y predicción del rendimiento de vídeos en YouTube España, mediante técnicas de *machine learning* supervisadas y no supervisadas, con el fin de identificar los factores cuantitativos y semánticos que determinan el **engagement** y la **visibilidad** dentro de la lógica algorítmica de la plataforma.

---

### Objetivos secundarios
1. **Extracción y estructuración del dataset**  
   Obtener y depurar un conjunto de datos representativo de vídeos de YouTube España a partir de la API v3, centrado en la categoría `mostPopular` y en contenidos con menos de 50.000 visualizaciones, con el fin de analizar el comportamiento de los vídeos emergentes.

2. **Extracción y estructuración del dataset**
   Preparación del dataset para su análisis estadístico y modelado, garantizando la coherencia, completitud y calidad de las variables.

3. **Exploración de variables estructurales y semánticas**  
   Analizar las relaciones entre las características de los vídeos (categoría, duración, etiquetas, descripciones, etc.) y sus métricas de rendimiento (views, likes, comentarios y engagement rate) mediante técnicas estadísticas y visualización de datos.

4. **Modelado predictivo supervisado**  
   Construir modelos de clasificación y regresión que permitan:
   - Estimar la probabilidad de que un vídeo alcance un alto nivel de engagement (cuartil superior).  
   - Predecir el número esperado de visualizaciones (*views*).

5. **Identificación de patrones y comunidades mediante aprendizaje no supervisado**  
   Implementar técnicas de reducción de dimensionalidad (*PCA, UMAP*) y *clustering* para detectar patrones latentes y comunidades de contenido, identificando posibles nichos temáticos o comportamientos algorítmicos.

6. **Interpretación teórico-práctica en el marco de la sociedad plataforma**  
   Analizar los resultados de los modelos y clusters desde los marcos conceptuales de la *sociedad plataforma* y la *curaduría algorítmica*, relacionando las dinámicas de visibilidad y monetización con los procesos de mediación tecnológica de YouTube.

---

## Marco teórico aplicado al análisis

El proyecto integra los resultados empíricos con las principales teorías contemporáneas sobre plataformas digitales:

| Enfoque teórico | Autores clave | Aplicación en el proyecto |
|------------------|---------------|----------------------------|
| **Sociedad plataforma** | Poell, Nieborg & Van Dijck (2019) | Explica cómo YouTube actúa como infraestructura socio-técnica y económica de visibilidad. |
| **Curaduría algorítmica** | Gillespie (2018), Jenkins (2006) | Interpreta los patrones de clustering y visibilidad como efectos de mediación algorítmica. |
| **Economía de la atención** | Davenport & Beck, Goldhaber | Reinterpreta las métricas de rendimiento como indicadores de valor atencional. |
| **Lógica de monetización y extracción de datos** | Srnicek (2017), Zuboff (2019) | Relaciona los patrones detectados con la economía política del dato y la visibilidad. |

---

## Elementos prácticos que sustentan el análisis

- **Resultados cuantitativos interpretables:**  
  - Feature importance de los modelos predictivos.  
  - Distribución de categorías y etiquetas por clusters.  
  - Comparativas entre vídeos emergentes y de alta visibilidad.  

- **Visualizaciones clave:**  
  - Mapas de clusters (UMAP/PCA) con métricas superpuestas.  
  - Gráficos de importancia de variables y tasas de engagement.  
  - Diagramas de comunidades temáticas.  

- **Discusión crítica:**  
  - Evaluación de sesgos algorítmicos y limitaciones del dataset.  
  - Reflexión sobre cómo los algoritmos moldean la producción y consumo de contenido audiovisual.

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
