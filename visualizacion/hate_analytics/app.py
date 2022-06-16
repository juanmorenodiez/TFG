from tokenize import Token
from unicodedata import name
import dash
import dash_core_components as dcc
import dash_html_components as html
from matplotlib.pyplot import title
from plotly.subplots import make_subplots
import pandas as pd
import dash.dependencies as dd
import plotly.graph_objects as go
from io import BytesIO
import base64
import plotly.express as px
import functions

# source TFGenv/bin/activate
# http://localhost:8050/
# https://plotly.com/python/histograms/
# https://plotly.com/python/bar-charts/


dtype={'user_id': float, 'verificado': bool, 'profile_changed': bool, 'img_profile_changed': bool}

dataframe = pd.read_csv("data/todos_tweets_juntos_labeled.csv")
dataframe = dataframe.drop(labels=910, axis=0)
dataframe = dataframe.drop(labels=29687, axis=0)


app = dash.Dash(__name__)



app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Hate Speech Analytics", className="header-title"
                ),
                html.P(
                    children="Hate Speech Analytics and Visualization system",
                    className="header-description"
                )
                
            ],
            className="header"
        ),

        dcc.Graph(id="subplots"),

        dcc.Graph(id="bar_chart_2"),

        dcc.Graph(id="bar_chart_4"),

        dcc.Graph(id="bar_chart"),

        dcc.Graph(id="bar_chart_1"),

        dcc.Graph(id="bar_chart_3"),

        html.Img(id="image_wc",
                title="Nube de Palabras",
                className="wordcloud"),   

        dcc.Graph(id="pie_chart"), 
        dcc.Graph(id="pie_chart_2"),
    ]
)

@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    functions.generate_wordcloud_experiment(dataframe, 15).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(dd.Output('bar_chart', 'figure'), [dd.Input('bar_chart', 'id')])
def update_bar_chart(day):
    comunidades, proporciones = functions.histogram_all_communities(dataframe, 5)
    fig = px.bar(x=comunidades, y=proporciones, title="Top 5 comunidades autónomas con proporcion mayor de usuarios haters en los tweets extraidos")
    functions.experimentos_tweets_odio(dataframe)
    return fig

@app.callback(dd.Output('pie_chart', 'figure'), [dd.Input('pie_chart', 'id')])
def update_bar_chart(day):
    tipos, cantidades = functions.experimentos_tweets(dataframe)
    fig = px.pie(values=cantidades, names=tipos, title='Porcentaje de tweets totales obtenidos por cada experimento')
    return fig

@app.callback(dd.Output('pie_chart_2', 'figure'), [dd.Input('pie_chart_2', 'id')])
def update_bar_chart(day):
    tipos, cantidades = functions.experimentos_tweets_odio(dataframe)
    fig = px.pie(values=cantidades, names=tipos, title='Porcentaje de tweets de odio totales obtenidos por cada experimento')
    return fig

@app.callback(dd.Output('bar_chart_1', 'figure'), [dd.Input('bar_chart_1', 'id')])
def update_bar_chart_1(day1):
    fig = go.Figure()

    tipos, frecuencias_hater = functions.histogram_verificados_haters(dataframe)
    frecuencias_all = functions.histogram_verificados_all(dataframe)
    frecuencias_hater_str = list(map(str, frecuencias_hater))
    frecuencias_all_str = list(map(str, frecuencias_all))

    tipos = ["Usuarios Haters", "Total Usuarios"]

    fig.add_trace(go.Histogram(y=frecuencias_hater_str, x=tipos,
                            histfunc="sum", name="Verificados"))
    fig.add_trace(go.Histogram(y=frecuencias_all_str, x=tipos,
                            histfunc="sum", name="No verificados"))

    fig.update_layout(title_text="Proporción verificados | no-verificados entre los usuarios haters y el total")

    return fig

@app.callback(dd.Output('bar_chart_2', 'figure'), [dd.Input('bar_chart_2', 'id')])
def update_bar_chart_2(day2):
    tipo, count = functions.get_interacciones_tweet_respuesta(dataframe)
    fig = px.bar(x=tipo, y=count, title="¿Qué suele hacer más un Hater? ¿Poner Tweets, Citar, Responder o Retuitear?")
    return fig

@app.callback(dd.Output('bar_chart_3', 'figure'), [dd.Input('bar_chart_3', 'id')])
def update_bar_chart_3(day3):
    nombres_usuario, seguidores = functions.most_popular_user_haters(dataframe, 5)
    fig = px.bar(x=nombres_usuario, y=seguidores, title="Top 5 usuarios haters con más seguidores")
    return fig

@app.callback(dd.Output('bar_chart_4', 'figure'), [dd.Input('bar_chart_4', 'id')])
def update_bar_chart_2(day4):
    hashtags, frecuencias = functions.most_popular_hashtags(dataframe, 10)
    fig = px.bar(x=hashtags, y=frecuencias, title="Top 10 hashtags más utilizados por usuarios haters")
    return fig

@app.callback(dd.Output('subplots', 'figure'), [dd.Input('subplots', 'id')])
def update_subplots(day4):
    fig = make_subplots(rows=1, cols=2)

    nombres_interacciones, interacciones = functions.interacciones_top_usuarios(dataframe, 5)
    interacciones_str = list(map(str, interacciones))
    nombres_tweets, num_tweets = functions.tweets_top_usuarios(dataframe, 5)
    num_tweets_str = list(map(str, num_tweets))

    trace0 = go.Histogram(y=interacciones_str, x=nombres_interacciones,
                            histfunc="sum", name="Usuarios con más Interacciones")
    trace1 = go.Histogram(y=num_tweets_str, x=nombres_tweets,
                            histfunc="sum", name="Usuarios con más Tweets")
    
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)

    fig.update_layout(title_text="Top 5 usuarios Hater que más Tuitean y que más interaccionan (Retuits + Favoritos)")

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)