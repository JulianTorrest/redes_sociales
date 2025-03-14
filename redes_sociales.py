import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Descargar datos de NLTK al inicio
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# URL del archivo CSV en GitHub
url_hoja_calculo = "https://raw.githubusercontent.com/JulianTorrest/redes_sociales/main/Redes%20Sociales.csv"

def leer_csv_desde_github(url):
    """Lee un archivo CSV desde una URL de GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo CSV: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error al analizar el archivo CSV: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None

def extraer_contenido_web(url):
    """Extrae el contenido de texto de una página web, omitiendo errores 403."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.warning(f"Acceso prohibido (403) a: {url}. Omitiendo.")
            return ""  # Devuelve una cadena vacía en caso de error 403
        else:
            st.error(f"Error al obtener la página web: {e}")
            return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener la página web: {e}")
        return ""
    except Exception as e:
        st.error(f"Error inesperado al extraer contenido web: {e}")
        return ""

def generar_nube_palabras(texto):
    """Genera una nube de palabras y la devuelve como imagen."""
    if texto:
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            image = BytesIO()
            plt.savefig(image, format='png')
            plt.close()
            return image
        except ValueError as e:
            st.error(f"Error generating wordcloud: {e}")
            return None
    else:
        st.warning("Warning: Empty text, skipping word cloud generation.")
        return None

def analizar_sentimientos_emociones_es(df):
    """Realiza análisis de sentimientos y emociones en español."""
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    if not df.empty:
        if 'Paginas relacionadas' in df.columns: # Corrección del nombre de la columna
            sentimientos_vader_publicacion = []
            sentimientos_textblob_publicacion = []
            sentimientos_vader_web = []
            sentimientos_textblob_web = []
            pertinencia = []
            reacciones_positivas = []
            for index, row in df.iterrows():
                publicacion = row['Publicación']
                url_web = row['Paginas relacionadas'] # Corrección del nombre de la columna
                reaccion = row.get("Reacciones Positivas", 0)
                reacciones_positivas.append(reaccion)
                try:
                    traduccion_publicacion = translator.translate(publicacion, dest='en').text
                    sentimiento_vader_publicacion = analyzer.polarity_scores(traduccion_publicacion)
                except:
                    sentimiento_vader_publicacion = {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
                sentimientos_vader_publicacion.append(sentimiento_vader_publicacion)
                sentimiento_textblob_publicacion = TextBlob(publicacion).sentiment.polarity
                sentimientos_textblob_publicacion.append(sentimiento_textblob_publicacion)
                contenido_web = extraer_contenido_web(url_web) if pd.notna(url_web) else ""
                if contenido_web:
                    try:
                        traduccion_web = translator.translate(contenido_web, dest='en').text
                        sentimiento_vader_web = analyzer.polarity_scores(traduccion_web)
                    except:
                        sentimiento_vader_web = {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
                    sentimientos_vader_web.append(sentimiento_vader_web)
                    sentimiento_textblob_web = TextBlob(contenido_web).sentiment.polarity
                    sentimientos_textblob_web.append(sentimiento_textblob_web)
                    pertinencia.append(1 if any(word in contenido_web for word in publicacion.split()) else 0)

                    # Nubes de palabras dentro de streamlit.
                    publicacion_wordcloud = generar_nube_palabras(publicacion)
                    if publicacion_wordcloud:
                      st.image(publicacion_wordcloud, caption=f"Nube de palabras para publicación {index}")

                    web_wordcloud = generar_nube_palabras(contenido_web)
                    if web_wordcloud:
                      st.image(web_wordcloud, caption=f"Nube de palabras para web {index}")

                else:
                    sentimientos_vader_web.append(None)
                    sentimientos_textblob_web.append(None)
                    pertinencia.append(None)
                # Depuración: imprimir la lista en cada iteración
                print("Sentimientos TextBlob Publicación:", sentimientos_textblob_publicacion)

            # Agregar depuración
            print("Longitud de sentimientos_textblob_publicacion:", len(sentimientos_textblob_publicacion))
            print("Primeros 5 elementos de sentimientos_textblob_publicacion:", sentimientos_textblob_publicacion[:5])

            df['Sentimientos_VADER_Publicacion'] = sentimientos_vader_publicacion
            df['Sentimientos_TextBlob_Publicacion'] = sentimientos_textblob_publicacion
            df['Sentimientos_VADER_Web'] = sentimientos_vader_web
            df['Sentimientos_TextBlob_Web'] = sentimientos_textblob_web
            df['Pertinencia'] = pertinencia
            df['Reacciones_Positivas'] = reacciones_positivas
            st.success("Análisis de sentimientos y emociones completado.")
        else:
            st.warning("La columna 'Paginas relacionadas' no se encuentra en el DataFrame.") # Corrección del nombre de la columna
    else:
        st.warning("El DataFrame está vacío.")

def generar_tablas_graficos(df):
    """Genera tablas y gráficos para cada red social."""
    if not df.empty:
        if len(df) > 0:
            # Tabla de resultados
            st.write("Tabla de resultados:")
            st.write(df.describe())
            # Gráficos de sentimientos
            plt.figure(figsize=(10, 5))
            plt.plot(df['Sentimientos_TextBlob_Publicacion'], label='Sentimientos TextBlob Publicación')
            plt.plot(df['Sentimientos_TextBlob_Web'], label='Sentimientos TextBlob Web')
            plt.title('Análisis de sentimientos')
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("El DataFrame está vacío, no se pueden generar tablas ni gráficos.")

def main():
    st.title("Análisis de Redes Sociales")
    dataframe = leer_csv_desde_github(url_hoja_calculo)
    if dataframe is not None:
        print(dataframe.head())  # Depuración: imprimir el DataFrame cargado
        analizar_sentimientos_emociones_es(dataframe)
        generar_tablas_graficos(dataframe)

if __name__ == "__main__":
    main()

