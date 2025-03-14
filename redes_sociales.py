import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt')

# URL del archivo CSV en GitHub
url_hoja_calculo = "https://raw.githubusercontent.com/JulianTorrest/redes_sociales/main/Redes%20Sociales.csv"

def leer_csv_desde_github(url):
    """
    Lee un archivo CSV desde una URL de GitHub y lo carga en un DataFrame.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo CSV: {e}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error al analizar el archivo CSV: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

def extraer_contenido_web(url):
    """Extrae el contenido de texto de una página web."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la página web: {e}")
        return ""
    except Exception as e:
        print(f"Error inesperado al extraer contenido web: {e}")
        return ""

# Leer el archivo CSV y obtener el DataFrame
dataframe = leer_csv_desde_github(url_hoja_calculo)

# Imprimir el DataFrame
if dataframe is not None:
    print("DataFrame:")
    print(dataframe.head())
    print("-" * 20)

    #Ejemplo de como usar la funcion extraer_contenido_web con una columna del dataframe.
    if 'URL' in dataframe.columns:
      for url in dataframe['URL'].head(): #Se usa head para solo tomar los primeros 5 urls.
        contenido = extraer_contenido_web(url)
        print(f"Contenido de {url}:")
        print(contenido[:200]) #Se imprimen los primeros 200 caracteres.
        print("-" * 20)
    else:
      print("La columna 'URL' no existe en el DataFrame.")

def generar_nube_palabras(texto, nombre_archivo):
    """Genera una nube de palabras y la guarda como imagen."""
    if texto:  # Verifica si el texto no está vacío
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(nombre_archivo)
            plt.close()
        except ValueError as e:
            print(f"Error generating wordcloud: {e}")
    else:
        print(f"Warning: Empty text, skipping word cloud generation for {nombre_archivo}.")

def analizar_sentimientos_emociones_es(dataframes):
    """Realiza análisis de sentimientos y emociones en español."""
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    for nombre_hoja, df in dataframes.items():
        if not df.empty: # Se remueve la condición and nombre_hoja != 'Linkedin'
            sentimientos_vader_publicacion = []
            sentimientos_textblob_publicacion = []
            sentimientos_vader_web = []
            sentimientos_textblob_web = []
            pertinencia = []
            reacciones_positivas = []
            for index, row in df.iterrows():
                publicacion = row['Publicación']
                url_web = row['Paginas relacionadas']
                reaccion = row.get("Reacciones Positivas", 0)
                reacciones_positivas.append(reaccion)
                # Análisis de la publicación
                try:
                    traduccion_publicacion = translator.translate(publicacion, dest='en').text
                    sentimiento_vader_publicacion = analyzer.polarity_scores(traduccion_publicacion)
                except:
                    sentimiento_vader_publicacion = {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
                sentimientos_vader_publicacion.append(sentimiento_vader_publicacion)
                sentimiento_textblob_publicacion = TextBlob(publicacion).sentiment.polarity
                sentimientos_textblob_publicacion.append(sentimiento_textblob_publicacion)
                # Análisis de la página web
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
                    # Nubes de palabras
                    generar_nube_palabras(publicacion, f"{nombre_hoja}_publicacion_{index}.png")
                    generar_nube_palabras(contenido_web, f"{nombre_hoja}_web_{index}.png")
                else:
                    sentimientos_vader_web.append(None)
                    sentimientos_textblob_web.append(None)
                    pertinencia.append(None)

            df['Sentimientos_VADER_Publicacion'] = sentimientos_vader_publicacion
            df['Sentimientos_TextBlob_Publicacion'] = sentimientos_textblob_publicacion
            df['Sentimientos_VADER_Web'] = sentimientos_vader_web
            df['Sentimientos_TextBlob_Web'] = sentimientos_textblob_web
            df['Pertinencia'] = pertinencia
            df['Reacciones_Positivas'] = reacciones_positivas
            print(f"Análisis completado para '{nombre_hoja}'.")
        else:
            print (f"DataFrame '{nombre_hoja}' esta vacio")

def generar_tablas_graficos(dataframes):
    """Genera tablas y gráficos para cada red social."""
    for nombre_hoja, df in dataframes.items():
        if not df.empty: # Se remueve la condición and nombre_hoja != 'Linkedin'
            if len(df) > 0: #Agregado, si el dataframe no tiene contenido, no genera nada.
                # Tabla de resultados
                print(f"Tabla de resultados para '{nombre_hoja}':")
                print(df.describe())
                # Gráficos de sentimientos
                plt.figure(figsize=(10, 5))
                plt.plot(df['Sentimientos_TextBlob_Publicacion'], label='Sentimientos TextBlob Publicación')
                plt.plot(df['Sentimientos_TextBlob_Web'], label='Sentimientos TextBlob Web')
                plt.title(f'Análisis de sentimientos para {nombre_hoja}')
                plt.legend()
                plt.show()
            else:
                print(f"Warning: DataFrame '{nombre_hoja}' is empty, skipping table and chart generation.")


# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Realizar análisis de sentimientos y emociones en español
if dataframes:
    analizar_sentimientos_emociones_es(dataframes)
    generar_tablas_graficos(dataframes)

import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from googletrans import Translator
import nltk
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

nltk.download('punkt')

def leer_gsheet_a_dataframes(url_hoja_calculo):
    """
    Lee un archivo .gsheet desde Google Drive y carga cada hoja en un DataFrame.
    """
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(url_hoja_calculo)
        dataframes = {}
        for worksheet in sh.worksheets():
            data = worksheet.get_all_values()
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            dataframes[worksheet.title] = df
        return dataframes
    except Exception as e:
        print(f"Error al leer el archivo .gsheet: {e}")
        return None

def extraer_contenido_web(url):
    """Extrae el contenido de texto de una página web."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except:
        return ""

def analizar_sentimientos_mensual(dataframes):
    """Realiza análisis de sentimientos mensual."""
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    for nombre_hoja, df in dataframes.items():
        if not df.empty and 'Fecha' in df.columns:
            try:
                if nombre_hoja == 'Linkedin':
                    # Asegurarse de que el formato de la fecha sea correcto
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%m/%Y')
                    df['Mes'] = df['Fecha'].dt.to_period('M')
                    resultados_mensuales = []
                    for mes, grupo in df.groupby('Mes'):
                        sentimientos_vader = []
                        for publicacion in grupo['Publicación'].dropna():
                            try:
                                traduccion_publicacion = translator.translate(publicacion, dest='en').text
                                sentimiento_vader = analyzer.polarity_scores(traduccion_publicacion)
                                sentimientos_vader.append(sentimiento_vader['compound'])
                            except:
                                sentimientos_vader.append(0)
                        promedio_vader = sum(sentimientos_vader) / len(sentimientos_vader) if sentimientos_vader else 0
                        resultados_mensuales.append({'Mes': mes, 'Promedio_VADER': promedio_vader})
                    resultados_df = pd.DataFrame(resultados_mensuales)
                    print(f"Análisis mensual para '{nombre_hoja}':\n{resultados_df}")
                    # Graficar resultados
                    plt.figure(figsize=(12, 6))
                    plt.plot(resultados_df['Mes'].astype(str), resultados_df['Promedio_VADER'], marker='o')
                    plt.title(f'Análisis de Sentimiento Mensual para {nombre_hoja}')
                    plt.xlabel('Mes')
                    plt.ylabel('Promedio Sentimiento VADER')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
                else:
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
                    df['Semana'] = df['Fecha'].dt.to_period('W')
                    resultados_semanales = []
                    for semana, grupo in df.groupby('Semana'):
                        sentimientos_vader = []
                        for publicacion in grupo['Publicación'].dropna():
                            try:
                                traduccion_publicacion = translator.translate(publicacion, dest='en').text
                                sentimiento_vader = analyzer.polarity_scores(traduccion_publicacion)
                                sentimientos_vader.append(sentimiento_vader['compound'])
                            except:
                                sentimientos_vader.append(0)
                        promedio_vader = sum(sentimientos_vader) / len(sentimientos_vader) if sentimientos_vader else 0
                        resultados_semanales.append({'Semana': semana, 'Promedio_VADER': promedio_vader})
                    resultados_df = pd.DataFrame(resultados_semanales)
                    print(f"Análisis semanal para '{nombre_hoja}':\n{resultados_df}")
                    # Graficar resultados
                    plt.figure(figsize=(12, 6))
                    plt.plot(resultados_df['Semana'].astype(str), resultados_df['Promedio_VADER'], marker='o')
                    plt.title(f'Análisis de Sentimiento Semanal para {nombre_hoja}')
                    plt.xlabel('Semana')
                    plt.ylabel('Promedio Sentimiento VADER')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()
            except ValueError:
                print(f"Error: La columna 'Fecha' no tiene el formato esperado en '{nombre_hoja}'.")
        else:
            print(f"DataFrame '{nombre_hoja}' esta vacio o no tiene columna 'Fecha'.")

# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Realizar análisis de sentimientos mensual
if dataframes:
    analizar_sentimientos_mensual(dataframes)

import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from IPython.display import Image, display

def leer_gsheet_a_dataframes(url_hoja_calculo):
    """
    Lee un archivo .gsheet desde Google Drive y carga cada hoja en un DataFrame.
    """
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(url_hoja_calculo)
        dataframes = {}
        for worksheet in sh.worksheets():
            data = worksheet.get_all_values()
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            dataframes[worksheet.title] = df
        return dataframes
    except Exception as e:
        print(f"Error al leer el archivo .gsheet: {e}")
        return None

def generar_nube_palabras(texto, nombre_archivo):
    """Genera una nube de palabras y la guarda como imagen."""
    if texto:
        try:
            stopwords = set(STOPWORDS)
            stopwords.update(["que", "de", "https", "bit.ly", "tu", "bit", "ly", "tus", "este", "solo", "puede", "junto", "Hoy", "aquí", "dia", "y", "aqui", "tus", "te", "la", "el", "los", "las", "en", "un", "una", "por", "para", "con", "sin", "del", "al", "se", "lo", "le", "su", "sus", "como", "más", "pero", "muy", "mucho", "donde", "cuando", "también", "desde", "hasta", "sobre", "entre", "si", "no", "es", "son", "está", "están", "fue", "han", "hay", "ya", "así", "esto", "ese", "esa", "esos", "esas", "aqui", "ahi", "allí", "mismo", "misma", "mismos", "mismas", "otro", "otra", "otros", "otras"])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(nombre_archivo)
            plt.close()
            display(Image(filename=nombre_archivo))
        except ValueError as e:
            print(f"Error generating wordcloud: {e}")
    else:
        print(f"Warning: Empty text, skipping word cloud generation for {nombre_archivo}.")

def generar_nubes_palabras_por_red_social(dataframes):
    """Genera nubes de palabras por red social."""
    for nombre_hoja, df in dataframes.items():
        if not df.empty:  # Eliminar la condición 'and nombre_hoja != 'Linkedin''
            textos = ' '.join(df['Publicación'].dropna().astype(str))
            if textos:
                generar_nube_palabras(textos, f"{nombre_hoja}_nube_palabras.png")
                print(f"Nube de palabras generada para '{nombre_hoja}'.")
            else:
                print(f"Advertencia: No hay publicaciones para {nombre_hoja}.")
        else:
            print(f"DataFrame '{nombre_hoja}' esta vacio")

# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Generar nubes de palabras por red social
if dataframes:
    generar_nubes_palabras_por_red_social(dataframes)

import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from IPython.display import Image, display

def leer_gsheet_a_dataframes(url_hoja_calculo):
    """
    Lee un archivo .gsheet desde Google Drive y carga cada hoja en un DataFrame.
    """
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(url_hoja_calculo)
        dataframes = {}
        for worksheet in sh.worksheets():
            data = worksheet.get_all_values()
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            dataframes[worksheet.title] = df
        return dataframes
    except Exception as e:
        print(f"Error al leer el archivo .gsheet: {e}")
        return None

def generar_nube_palabras(texto, nombre_archivo):
    """Genera una nube de palabras y la guarda como imagen."""
    if texto:
        try:
            stopwords = set(STOPWORDS)
            stopwords.update(["que", "de", "https", "bit.ly", "tu", "bit", "ly", "este", "solo", "puede", "junto", "Hoy", "aquí", "dia", "y", "aqui", "tus", "te", "la", "el", "los", "las", "en", "un", "una", "por", "para", "con", "sin", "del", "al", "se", "lo", "le", "su", "sus", "como", "más", "pero", "muy", "mucho", "donde", "cuando", "también", "desde", "hasta", "sobre", "entre", "si", "no", "es", "son", "está", "están", "fue", "han", "hay", "ya", "así", "esto", "ese", "esa", "esos", "esas", "aqui", "ahi", "allí", "mismo", "misma", "mismos", "mismas", "otro", "otra", "otros", "otras"])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(nombre_archivo)
            plt.close()
            display(Image(filename=nombre_archivo))
        except ValueError as e:
            print(f"Error generating wordcloud: {e}")
    else:
        print(f"Warning: Empty text, skipping word cloud generation for {nombre_archivo}.")

def generar_nubes_palabras_por_mes_red_social(dataframes):
    """Genera nubes de palabras por mes y red social."""
    for nombre_hoja, df in dataframes.items():
        if not df.empty and nombre_hoja != 'Linkedin':
            if 'Fecha' in df.columns:
                try:
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
                    df['Mes'] = df['Fecha'].dt.to_period('M')
                    for mes, grupo in df.groupby('Mes'):
                        textos = ' '.join(grupo['Publicación'].dropna().astype(str))
                        if textos:
                            generar_nube_palabras(textos, f"{nombre_hoja}_{mes}_nube_palabras.png")
                            print(f"Nube de palabras generada para '{nombre_hoja}' en {mes}.")
                        else:
                            print(f"Advertencia: No hay publicaciones para {nombre_hoja} en {mes}.")
                except ValueError:
                    print(f"Error: La columna 'Fecha' no tiene el formato esperado en '{nombre_hoja}'.")
            else:
                print(f"Advertencia: La columna 'Fecha' no existe en '{nombre_hoja}'.")
        elif nombre_hoja == 'Linkedin':
            print("Linkedin esta vacio")
        else:
            print(f"DataFrame '{nombre_hoja}' esta vacio")

# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Generar nubes de palabras por mes y red social
if dataframes:
    generar_nubes_palabras_por_mes_red_social(dataframes)

import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from IPython.display import Image, display

def leer_gsheet_a_dataframes(url_hoja_calculo):
    """
    Lee un archivo .gsheet desde Google Drive y carga cada hoja en un DataFrame.
    """
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(url_hoja_calculo)
        dataframes = {}
        for worksheet in sh.worksheets():
            data = worksheet.get_all_values()
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            dataframes[worksheet.title] = df
        return dataframes
    except Exception as e:
        print(f"Error al leer el archivo .gsheet: {e}")
        return None

def generar_nube_palabras(texto, nombre_archivo):
    """Genera una nube de palabras y la guarda como imagen."""
    if texto:
        try:
            stopwords = set(STOPWORDS)
            stopwords.update(["que", "de", "https", "bit.ly", "tu", "bit", "ly", "este", "solo", "puede", "junto", "Hoy", "aquí", "dia", "y", "aqui", "tus", "te", "la", "el", "los", "las", "en", "un", "una", "por", "para", "con", "sin", "del", "al", "se", "lo", "le", "su", "sus", "como", "más", "pero", "muy", "mucho", "donde", "cuando", "también", "desde", "hasta", "sobre", "entre", "si", "no", "es", "son", "está", "están", "fue", "han", "hay", "ya", "así", "esto", "ese", "esa", "esos", "esas", "aqui", "ahi", "allí", "mismo", "misma", "mismos", "mismas", "otro", "otra", "otros", "otras"])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(nombre_archivo)
            plt.close()
            display(Image(filename=nombre_archivo))
        except ValueError as e:
            print(f"Error generating wordcloud: {e}")
    else:
        print(f"Warning: Empty text, skipping word cloud generation for {nombre_archivo}.")

def generar_nubes_palabras_por_mes_red_social(dataframes):
    """Genera nubes de palabras por mes y red social."""
    for nombre_hoja, df in dataframes.items():
        if not df.empty: # Se remueve la condición and nombre_hoja != 'Linkedin'
            if 'Fecha' in df.columns:
                try:
                    if nombre_hoja == 'Linkedin':
                        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%m/%Y') # Formato para LinkedIn
                    else:
                        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
                    df['Mes'] = df['Fecha'].dt.to_period('M')
                    for mes, grupo in df.groupby('Mes'):
                        textos = ' '.join(grupo['Publicación'].dropna().astype(str))
                        if textos:
                            generar_nube_palabras(textos, f"{nombre_hoja}_{mes}_nube_palabras.png")
                            print(f"Nube de palabras generada para '{nombre_hoja}' en {mes}.")
                        else:
                            print(f"Advertencia: No hay publicaciones para {nombre_hoja} en {mes}.")
                except ValueError:
                    print(f"Error: La columna 'Fecha' no tiene el formato esperado en '{nombre_hoja}'.")
            else:
                print(f"Advertencia: La columna 'Fecha' no existe en '{nombre_hoja}'.")
        else:
            print(f"DataFrame '{nombre_hoja}' esta vacio")

# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Generar nubes de palabras por mes y red social
if dataframes:
    generar_nubes_palabras_por_mes_red_social(dataframes)

import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from googletrans import Translator
import nltk
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))

def leer_gsheet_a_dataframes(url_hoja_calculo):
    """
    Lee un archivo .gsheet desde Google Drive y carga cada hoja en un DataFrame.
    """
    try:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(url_hoja_calculo)
        dataframes = {}
        for worksheet in sh.worksheets():
            data = worksheet.get_all_values()
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            dataframes[worksheet.title] = df
        return dataframes
    except Exception as e:
        print(f"Error al leer el archivo .gsheet: {e}")
        return None

def extraer_contenido_web(url):
    """Extrae el contenido de texto de una página web."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except:
        return ""

def preprocesar_texto(texto):
    """Preprocesa el texto para NLP."""
    tokens = nltk.word_tokenize(texto.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def generar_nube_palabras(texto, nombre_archivo, stopwords=None):
    """Genera una nube de palabras y la guarda como imagen."""
    if texto:
        try:
            if stopwords is None:
                stopwords = set(STOPWORDS)
                stopwords.update(["que", "de", "https", "bit.ly", "tu", "bit", "ly", "este", "solo", "puede", "junto", "Hoy", "aquí", "dia", "y", "aqui", "tus", "te", "la", "el", "los", "las", "en", "un", "una", "por", "para", "con", "sin", "del", "al", "se", "lo", "le", "su", "sus", "como", "más", "pero", "muy", "mucho", "donde", "cuando", "también", "desde", "hasta", "sobre", "entre", "si", "no", "es", "son", "está", "están", "fue", "han", "hay", "ya", "así", "esto", "ese", "esa", "esos", "esas", "aqui", "ahi", "allí", "mismo", "misma", "mismos", "mismas", "otro", "otra", "otros", "otras"])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(nombre_archivo)
            plt.close()
        except ValueError as e:
            print(f"Error generating wordcloud: {e}")
    else:
        print(f"Warning: Empty text, skipping word cloud generation for {nombre_archivo}.")

def analizar_sentimientos_y_clusters(dataframes):
    """Realiza análisis de sentimientos, clusters y nubes de palabras."""
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    for nombre_hoja, df in dataframes.items():
        if not df.empty and 'Fecha' in df.columns:
            try:
                if nombre_hoja == 'Linkedin':
                    # Modificado para manejar el formato MM/YYYY
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%m/%Y')
                    df['Mes'] = df['Fecha'].dt.to_period('M')
                    resultados_mensuales = []
                    textos_positivos = []
                    textos_negativos = []
                    for mes, grupo in df.groupby('Mes'):
                        sentimientos_vader = []
                        textos_preprocesados = [preprocesar_texto(publicacion) for publicacion in grupo['Publicación'].dropna()]
                        for publicacion in grupo['Publicación'].dropna():
                            try:
                                traduccion_publicacion = translator.translate(publicacion, dest='en').text
                                sentimiento_vader = analyzer.polarity_scores(traduccion_publicacion)
                                sentimientos_vader.append(sentimiento_vader['compound'])
                                if sentimiento_vader['compound'] > 0:
                                    textos_positivos.append(publicacion)
                                else:
                                    textos_negativos.append(publicacion)
                            except:
                                sentimientos_vader.append(0)
                        promedio_vader = sum(sentimientos_vader) / len(sentimientos_vader) if sentimientos_vader else 0
                        resultados_mensuales.append({'Mes': mes, 'Promedio_VADER': promedio_vader, 'Textos_Preprocesados': ' '.join(textos_preprocesados)})
                    resultados_df = pd.DataFrame(resultados_mensuales)
                    print(f"Análisis mensual para '{nombre_hoja}':\n{resultados_df}")
                else:
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
                    df['Semana'] = df['Fecha'].dt.to_period('W')
                    resultados_semanales = []
                    textos_positivos = []
                    textos_negativos = []
                    for semana, grupo in df.groupby('Semana'):
                        sentimientos_vader = []
                        textos_preprocesados = [preprocesar_texto(publicacion) for publicacion in grupo['Publicación'].dropna()]
                        for publicacion in grupo['Publicación'].dropna():
                            try:
                                traduccion_publicacion = translator.translate(publicacion, dest='en').text
                                sentimiento_vader = analyzer.polarity_scores(traduccion_publicacion)
                                sentimientos_vader.append(sentimiento_vader['compound'])
                                if sentimiento_vader['compound'] > 0:
                                    textos_positivos.append(publicacion)
                                else:
                                    textos_negativos.append(publicacion)
                            except:
                                sentimientos_vader.append(0)
                        promedio_vader = sum(sentimientos_vader) / len(sentimientos_vader) if sentimientos_vader else 0
                        resultados_semanales.append({'Semana': semana, 'Promedio_VADER': promedio_vader, 'Textos_Preprocesados': ' '.join(textos_preprocesados)})
                    resultados_df = pd.DataFrame(resultados_semanales)
                    print(f"Análisis semanal para '{nombre_hoja}':\n{resultados_df}")

                # Nubes de palabras por sentimiento
                generar_nube_palabras(' '.join(textos_positivos), f"{nombre_hoja}_positivas_nube_palabras.png")
                generar_nube_palabras(' '.join(textos_negativos), f"{nombre_hoja}_negativas_nube_palabras.png")

                # Clusterización de palabras
                if not resultados_df.empty:
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(resultados_df['Textos_Preprocesados'])
                    kmeans = KMeans(n_clusters=3)
                    resultados_df['Cluster'] = kmeans.fit_predict(X)
                    print(f"Clusters para '{nombre_hoja}':\n{resultados_df[['Mes' if nombre_hoja == 'Linkedin' else 'Semana', 'Cluster']]}")

                    # Gráfico de clusters
                    plt.figure(figsize=(10, 6))
                    eje_x = resultados_df['Mes'].astype(str) if nombre_hoja == 'Linkedin' else resultados_df['Semana'].astype(str)
                    sns.scatterplot(x=range(len(eje_x)), y='Promedio_VADER', hue='Cluster', data=resultados_df, palette='viridis')
                    plt.xticks(range(len(eje_x)), eje_x, rotation=45, ha='right')
                    plt.title(f"Clusters para '{nombre_hoja}'")
                    plt.xlabel('Mes' if nombre_hoja == 'Linkedin' else 'Semana')
                    plt.ylabel('Promedio Sentimiento VADER')
                    plt.show()

            except ValueError:
                print(f"Error: La columna 'Fecha' no tiene el formato esperado en '{nombre_hoja}'.")
        else:
            print(f"DataFrame '{nombre_hoja}' esta vacio o no tiene columna 'Fecha'.")

# Leer el archivo y obtener los DataFrames
dataframes = leer_gsheet_a_dataframes(url_hoja_calculo)

# Realizar análisis de sentimientos, clusters y nubes de palabras
if dataframes:
    analizar_sentimientos_y_clusters(dataframes)
