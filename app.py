from flask import *
import pymongo
from bson.objectid import ObjectId
import requests
from bs4 import BeautifulSoup
import re
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


app = Flask(__name__, static_folder = 'public', static_url_path = '/')
app.secret_key = 'abcxyz'


### connect to MongoDB
### input the personal URI to connect to a MongoDB database hosted on the MongoDB Atlas cloud service
# uri = "The string is a connection URI (Uniform Resource Identifier)"

### create a MongoDB client object (client) using the pymongo library in Python
# client = pymongo.MongoClient(uri)

### create a variable named db to represent the MongoDB database (seo_system)
# db = client.seo_system

### creates a variable named collection to represent webtext (the name of a collection within the seo_system database)
# collection = db.webtext



# Functions 
# (for filter divs & find the text and the key word list & clean the key word list)
def filter_div(div):
    if div.find(id = "st-toggle"):
        return False
    if not div.text:
        return False
    if div.find("w-ad-creative-spacer"):
        return False
    return True

def clean(sentence_ws, sentence_pos):
    short_with_pos = []
    short_sentence = []

    # stop_pos means these categories are excluded
    stop_pos = set(['Nep', 'Nh', 'Nb', 'Neu', 'Nc', 'Nd'])

    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        # only keep N & V
        is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N")
        # delete words in stop_pos
        is_not_stop_pos = word_pos not in stop_pos
        # delete only one word
        is_not_one_charactor = not (len(word_ws) == 1)
        
        if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
            short_with_pos.append(f"{word_ws}({word_pos})")
            short_sentence.append(f"{word_ws}")

    return (" ".join(short_sentence), " ".join(short_with_pos))

def find_keyword_list(total_page, query, ws_driver, pos_driver, ner_driver):
        total_page = int(total_page)
        starts = [1+i*9 for i in range(total_page)]
        text = []
        text_2 = []
        link = []

        for start in starts:
            divs = []
            
            # set target url
            url = f"https://www.google.com/search?q={query}&gl=tw&start={start}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            
            soup = BeautifulSoup(response.text, 'lxml')

            # scrape each website link
            for r in soup.find_all("h3"):
                pattern = re.compile(r"(?<=\=)(.*?)(?=&)")
                link.append(re.search(pattern, r.find_parent("a", recursive=False)["href"]).group(0))

            # scrape the text and combine them into one list
            divs = soup.find(id="main").findChildren("div", recursive=False)
            if start == 1:
                text = [div.text for div in filter(filter_div, divs)]
            else:
                text_2 = [div.text for div in filter(filter_div, divs)]
                text += text_2[1:]

        # apply CKIP to find key words
        ws  = ws_driver(text)
        pos = pos_driver(ws)
        ner = ner_driver(text)

        keyword_list = []
        for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
            (short, res) = clean(sentence_ws, sentence_pos)
            keyword_list.extend(short.split(' '))
        
        return [text, keyword_list, link]


# set CKIP Drivers
ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
ner_driver = CkipNerChunker(model="bert-base")


# Route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods = ['GET', 'POST'])
def search():
    query = request.form.get("title")
    total_page = request.form.get("pages")

    # get query results
    text, keyword_list, link = find_keyword_list(total_page, query, ws_driver, pos_driver, ner_driver)

    # add to MongoDB
    new_text = {"title": query, 'pages': total_page, 'keyword_list': keyword_list}   
    collection.insert_one(new_text)

    # count key words in the whole 'keyword_list' & in each website 'text[i]' 
    value_counts = pd.Series(keyword_list).value_counts()
    df = pd.DataFrame({'Word': value_counts.index, 'Count': value_counts.values})
    sub_df = df.sort_values(by=['Count'], ascending=False).reset_index(drop=True).head(20)

    for i in range(len(text)):
        if i < 9:
            sub_df[f'count_0{i+1}'] = None
        else:
            sub_df[f'count_{i+1}'] = None
        for j in range(len(sub_df['Word'])):
            if i < 9:
                sub_df[f'count_0{i+1}'][j] = text[i].count(sub_df['Word'][j])
            else:
                sub_df[f'count_{i+1}'][j] = text[i].count(sub_df['Word'][j])

    # draw bar plot
    layout = go.Layout(title=f'關鍵字為 "{query}" 在Google的前{total_page}頁搜尋紀錄中 出現的單詞次數統計表')
    fig = go.Figure(data=go.Bar(x=sub_df['Word'], y=sub_df['Count'], marker_color='skyblue'), layout=layout)  
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    plot_html = fig.to_html()

    # Store plot file path and other variables in session
    plot_file = f"plot_{query}.html"
    with open(plot_file, "w") as f:
        f.write(plot_html)
    session['plot_file'] = plot_file
    sub_df_dict = sub_df.to_dict()
    session['sub_df'] = sub_df_dict
    session['link'] = link

    return render_template('index.html', plot_html=plot_html)


@app.route('/search2', methods = ['GET', 'POST'])
def search2():
    start = int(request.form.get("start"))
    end = int(request.form.get("end"))

    # get variables from session
    plot_file = session.get('plot_file', None)
    with open(plot_file, "r") as f:
            plot_html = f.read()
    link = session.get('link', None)
    sub_df_dict = session.get('sub_df', None)
    sub_df = pd.DataFrame.from_dict(sub_df_dict)

    # filter dataframe
    sub_df_show = sub_df.copy()
    sub_df_show = sub_df_show.iloc[:10, [1] + list(range(start+1, end+2))]
    showlink = []
    showlink.append('--')
    showlink += link[start:end+1]
    sub_df_show.loc['link'] = showlink
    transposed_df = sub_df_show.T

    table_html = transposed_df.to_html()

    return render_template('index.html', plot_html=plot_html, table_html=table_html)



app.run(port = 3000)