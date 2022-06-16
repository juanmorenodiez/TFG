import re
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import ast
import pandas as pd
import random
import spacy
from gensim.corpora import Dictionary
import gensim.models
from nltk.corpus import stopwords
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

# https://github.com/ArmandDS/topic_modeling/blob/master/topic_modeling_spanish.ipynb

nltk.download('stopwords')
nlp = spacy.load('es_core_news_sm')

black_list = ['más', 'mas', 'unir', 'paises', 'pais', 'espa', 'no', 'os', 'a', 'compa', 'acompa', 'off', 'and', 'grecia', 'the','it', 'to',
              'd',  'et',  'dame',  'il',  'dans', 'that',  'as',   'for',  'it',  'elections',  'would',  'this',  'with', 'york', 'obama', 'chavez', 'gadafi']

list_comunidades_autonomas = ['andalucia','aragon', 'asturias', 'baleares', 'canarias', 'cantabria', 'castilla leon', 'castilla la mancha',
                                'cataluña', 'valencia', 'galicia', 'madrid', 'extremadura', 'murcia', 'navarra', 'pais vasco', 'rioja', 'ceuta', 'melilla']

def experimentos_tweets(df):
    dict_tipos_experimentos = {}
    for index, row in df.iterrows():
        tipo_experimento = row["tipo_experimento"]

        if tipo_experimento not in dict_tipos_experimentos.keys():
            dict_tipos_experimentos[tipo_experimento] = 0
            dict_tipos_experimentos[tipo_experimento] += 1
        else:
            dict_tipos_experimentos[tipo_experimento] += 1
    
    return list(dict_tipos_experimentos.keys()), list(dict_tipos_experimentos.values())

def experimentos_tweets_odio(df):
    dict_tipos_experimentos = {}
    for index, row in df.iterrows():
        if row["label"] == 1:
            tipo_experimento = row["tipo_experimento"]

            if tipo_experimento not in dict_tipos_experimentos.keys():
                dict_tipos_experimentos[tipo_experimento] = 0
                dict_tipos_experimentos[tipo_experimento] += 1
            else:
                dict_tipos_experimentos[tipo_experimento] += 1
    
    return list(dict_tipos_experimentos.keys()), list(dict_tipos_experimentos.values())


def histogram_verificados_haters(df):
    dict_usuarios = {}
    
    dict_usuarios["Verificado"] = 0
    dict_usuarios["No Verificado"] = 0

    for index, row in df.iterrows():
        if row["is_hater"] == 1 and row["verificado"] == True:
            dict_usuarios["Verificado"] += 1
        elif row["is_hater"] == 1 and row["verificado"] == False:
            dict_usuarios["No Verificado"] += 1
    
    return list(dict_usuarios.keys()), list(dict_usuarios.values())

def histogram_verificados_all(df):
    dict_usuarios = {}
    
    dict_usuarios["Verificado"] = 0
    dict_usuarios["No Verificado"] = 0

    for index, row in df.iterrows():
        if row["verificado"] == True:
            dict_usuarios["Verificado"] += 1
        elif row["verificado"] == False:
            dict_usuarios["No Verificado"] += 1
    
    return list(dict_usuarios.values())

def histogram_all_communities(df, top):
    dict_localizaciones = {}
    for index, row in df.iterrows():
        localizacion_aux = row['localizacion'].lower()
        new_loc = ''
        if type(localizacion_aux) == str:
            for i, comunidad in enumerate(list_comunidades_autonomas):
                if comunidad in localizacion_aux:
                    new_loc = list_comunidades_autonomas[i]
                    break
            if row["is_hater"] == 1:
                if new_loc not in dict_localizaciones.keys():
                    dict_localizaciones[new_loc] = [0, 0]
                else:
                    dict_localizaciones[new_loc][0] += 1
            else:
                if new_loc not in dict_localizaciones.keys():
                    dict_localizaciones[new_loc] = [0, 0]
                else:
                    dict_localizaciones[new_loc][1] += 1
    
    dict_nuevo_localizaciones = {}
    for loc in dict_localizaciones.keys():
        if loc not in dict_nuevo_localizaciones.keys():
            try:
                proporcion = dict_localizaciones[loc][0] / (dict_localizaciones[loc][0]+dict_localizaciones[loc][1])
                if proporcion != 0.0 and proporcion < 1:
                    dict_nuevo_localizaciones[loc] = dict_localizaciones[loc][0] / (dict_localizaciones[loc][0]+dict_localizaciones[loc][1])
            except ZeroDivisionError as e:
                continue

    sorted_dict = sorted(dict_nuevo_localizaciones.items(), key=lambda kv: kv[1], reverse=True)
    comunidades = [tupla[0] for tupla in sorted_dict[:top]]
    proporciones_hate = [tupla[1] for tupla in sorted_dict[:top]]

    return comunidades, proporciones_hate

# tarta con la divison de tweets por cada experimento para ver donde se saca más
 
def generate_wordcloud_experiment(df, k):
  splitted_tweets = [tweet.split() for tweet in df.loc[df['is_hater'] == 1].text if type(tweet) == str]
  # aqui quedarme con nombres y adjetivos
  list_all_words = []
  for tweet in splitted_tweets:
      no_urls = [re.sub(r'http\S+', '', word) for word in tweet]
      text_processed = [re.sub(r"[^a-zA-Z0-9_ÑñÁáÉéÍíÓóÚú]+", '', word) for word in no_urls]
      for word in text_processed:
          if word != '' and word not in stopwords.words('spanish') and len(word) > 3:
            list_all_words.append(word)
  counter = Counter(list_all_words)
  most_occur = counter.most_common(k)
  wordcloud = WordCloud().generate_from_frequencies(dict(most_occur))

  return wordcloud.to_image()

def get_interacciones_tweet_respuesta(df):
    dict_interacciones = {}
    dict_interacciones["Responder"] = 0
    dict_interacciones["Citar"] = 0
    dict_interacciones["Retuitear"] = 0
    dict_interacciones["Tweet"] = 0

    for index, row in df.iterrows():
        if row["is_hater"] == 1:
            if row["is_reply"] == True:
                dict_interacciones['Responder'] += 1
            elif row["is_quote"] == True:
                dict_interacciones['Citar'] += 1
            elif row["is_rt"] == True:
                dict_interacciones['Retuitear'] += 1
            else:
                dict_interacciones['Tweet'] += 1

    return list(dict_interacciones.keys()), list(dict_interacciones.values())

def most_popular_user_haters(df, top):
    dict_tweets = {}

    for index, row in df.iterrows():
        if row["is_hater"] == 1:
            dict_tweets[row["screen_name"]] = row["followers_count"]
    
    dict_tweets_proccessed = {}
    for key in dict_tweets.keys():
        if type(dict_tweets[key]) == int:
            if key not in dict_tweets_proccessed.keys():
                dict_tweets_proccessed[key] = dict_tweets[key]

    sorted_dict = sorted(dict_tweets_proccessed.items(), key=lambda kv: kv[1], reverse=True)
    nombres_usuario = [tupla[0] for tupla in sorted_dict[:top]]
    seguidores = [tupla[1] for tupla in sorted_dict[:top]]

    return nombres_usuario, seguidores

def most_popular_hashtags(df, top):
    dict_hashtags = {}

    for index, row in df.iterrows():
        if row["is_hater"] == 1:
            dict_aux = ast.literal_eval(row["top_hashtags"])
            for hashtag in dict_aux.keys():
                if hashtag not in dict_hashtags.keys():
                    dict_hashtags[hashtag] = dict_aux[hashtag]
                else:
                    dict_hashtags[hashtag] += dict_aux[hashtag]

    sorted_dict = sorted(dict_hashtags.items(), key=lambda kv: kv[1], reverse=True)
    hashtags = [tupla[0] for tupla in sorted_dict[:top]]
    frecuencias = [tupla[1] for tupla in sorted_dict[:top]]

    return hashtags, frecuencias

def display_topics(model, model_type="lda"):
    for topic_idx, topic in enumerate(model.print_topics()):
        print ("Topic %d:" % (topic_idx))
        if model_type== "hdp":
            print (" ".join(re.findall( r'\*(.[^\*-S]+).?', topic[1])), "\n")
        else:
            print (" ".join(re.findall( r'\"(.[^"]+).?', topic[1])), "\n")

def top_topics(df):
    tweets = df['text'].to_list()

    cleaned_tweets = [cleaner(tweet) for tweet in tweets if cleaner(tweet) != '']
    dataset = [d.split() for d in cleaned_tweets]

    dictionary = Dictionary(dataset)
    dictionary.compactify()
    # Filter extremes
    dictionary.filter_extremes(no_below=2, no_above=0.97, keep_n=None)
    dictionary.compactify()

    corpus = [dictionary.doc2bow(text) for text in dataset]
    ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

    display_topics(ldamodel)

    df_topic_sents_keywords = format_topics_sentences(ldamodel, corpus, texts=dataset)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    print(df_dominant_topic.head(10))

    """
    label_dicc = {}

    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].astype('int64')
    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].map(label_dicc)
    df_dominant_topic.head(10)

    df['labels'] = df_dominant_topic['Dominant_Topic']
    df[['text', 'labels']].head(10)

    return df_dominant_topic['Dominant_Topic'].value_counts()"""



def lemmatization(texts, allowed_postags=['NOUN']):
    texts_out = [token.text for token in nlp(texts) if token.pos_ in 
                 allowed_postags and token.text not in black_list and len(token.text)>2]
    return texts_out

def cleaner(sentence):
    sentence = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z[){2-6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', sentence, flags=re.MULTILINE)
    sentence = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', "", sentence)
    sentence = re.sub(r'ee.uu', "eeuu", sentence)
    sentence = re.sub(r'\#\.', "", sentence)
    sentence = re.sub(r'\n', "", sentence)
    sentence = re.sub(r',', "", sentence)
    sentence = re.sub(r'\-', "", sentence)
    sentence = re.sub(r'\.{3}', "", sentence)
    sentence = re.sub(r'a{2,', "a", sentence)
    sentence = re.sub(r'é{2,', "é", sentence)
    sentence = re.sub(r'i{2,', "i", sentence)
    sentence = re.sub(r'ja{2,', "ja", sentence)
    sentence = re.sub(r'á', "a", sentence)
    sentence = re.sub(r'é', "e", sentence)
    sentence = re.sub(r'í', "i", sentence)
    sentence = re.sub(r'ó', "o", sentence)
    sentence = re.sub(r'ú', "u", sentence)
    sentence = re.sub(r'[^a-zA-Z]', " ", sentence)

    stop_words = set(stopwords.words("spanish"))
    list_word_clean = []

    for w1 in sentence.split(" "):
        if w1.lower() not in stop_words:
            list_word_clean.append(w1.lower())
    
    bigram = gensim.models.Phrases(list_word_clean)
    bigrams_list = bigram[list_word_clean]
    out_text = lemmatization(" ".join(bigrams_list))

    return " ".join(out_text)

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def interacciones_top_usuarios(df, top):
    dict_interacciones = {}

    for index, row in df.iterrows():
        if row["is_hater"] == 1:
            if row["screen_name"] not in dict_interacciones.keys():
                try:
                    dict_interacciones[row["screen_name"]] = row["favourites_count"] + row["rt_count"]
                except TypeError as e1:
                    continue
            else:
                try:
                    dict_interacciones[row["screen_name"]] = row["favourites_count"] + row["rt_count"]
                except TypeError as e2:
                    continue


    sorted_dict = sorted(dict_interacciones.items(), key=lambda kv: kv[1], reverse=True)
    nombres_usuarios = [tupla[0] for tupla in sorted_dict[:top]]
    interacciones = [tupla[1] for tupla in sorted_dict[:top]]

    return nombres_usuarios, interacciones

def tweets_top_usuarios(df, top):
    dict_tweets = {}

    for index, row in df.iterrows():
        if row["is_hater"] == 1:
            if row["screen_name"] not in dict_tweets.keys():
                dict_tweets[row["screen_name"]] = row["statuses_count"]
            else:
                dict_tweets[row["screen_name"]] = row["statuses_count"]

    dict_tweets_proccessed = {}
    for key in dict_tweets.keys():
        if type(dict_tweets[key]) == int:
            if key not in dict_tweets_proccessed.keys():
                dict_tweets_proccessed[key] = dict_tweets[key]

    sorted_dict = sorted(dict_tweets_proccessed.items(), key=lambda kv: kv[1], reverse=True)
    nombres_usuarios = [tupla[0] for tupla in sorted_dict[:top]]
    tweets = [tupla[1] for tupla in sorted_dict[:top]]

    return nombres_usuarios, tweets



