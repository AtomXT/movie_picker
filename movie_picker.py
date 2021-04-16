# -*- coding:utf-8 -*-
# author:TongXu
# datetime:2021/4/2 22:55

from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import sqlite3
import numpy as np
import os
from flask import Flask, request, render_template

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_SOURCE = 'https://www.imdb.com/search/title/?title_type=feature&sort=num_votes,desc'
CACHE_FILE_NAME = 'movies_cache.json'
DATABASE_NAME = 'movies_db.sqlite'
WEBSITE_URL = 'https://www.imdb.com'
INFORMATION_LIST = ['name', 'genres', 'link', 'runtime', 'certificate', 'rating', 'year', 'votes', 'director', 'poster']
GENRES = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
       'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
       'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport',
       'Thriller', 'War', 'Western']
TOTAL_MOVIES = 10000  # After 10000, the anti-crawler protection of this website will encrypt the URL,
# so I only fetch 10000 movies. Since I sort all movies by the number of votes,
# it's reasonable for me to just use the first 10000 movies. After 10000, only few people have seen it,
# which means it may not be a good choice for users.


def load_cache():
    """
    Load cache from local. If it doesn't exist, set as an empty set.

    Returns:
    -------
    Dict
        Raw result from scraping.
    """
    try:
        cache_file = open(CACHE_FILE_NAME, 'r')
        cache_file_contents = cache_file.read()
        cache = json.loads(cache_file_contents)
        cache_file.close()
    except:
        cache = {}
    return cache


def save_cache(cache):
    """
    Save cache to local file.

    Parameters
    ----------
    cache : set
        Cache data to save.
    """
    cache_file = open(CACHE_FILE_NAME, 'w')
    contents_to_write = json.dumps(cache)
    cache_file.write(contents_to_write)
    cache_file.close()


def make_url_request_using_cache(url, cache):
    """
    Load cache from local. If it doesn't exist, set as an empty set.

    Parameters
    ----------
    url : string
        URL for scraping.
    cache : dict
        Cache data for scraping.

    Returns
    -------
    String
        Raw result from scraping the specified url.
    """
    if url in cache.keys():  # the url is our unique key
        print("Using cache.")
        return cache[url]
    else:
        print("Fetching...")
        # time.sleep(0.1)
        response = requests.get(url)
        cache[url] = response.text
        save_cache(cache)
        return cache[url]


def fetch_data_of_single_page(single_page):
    """
    Get movie data from a single page.

    Parameters
    ----------
    single_page : string
        Raw result from scraping a single page.

    Returns
    -------
    data : list
        List of all tags. Each of them is a movie.
    """
    single_soup = BeautifulSoup(single_page, 'html.parser')
    datas = []
    movies = single_soup.findAll('div', class_="lister-item mode-advanced")
    for movie in movies:
        datas.append(extract_data_of_single_movie(movie))
    return datas


def extract_data_of_single_movie(tag):
    """
    Extract data of a single movie.

    Parameters
    ----------
    tag : bs4.element.Tag

    Returns
    -------
    List
        All information we needed for a single movie.
    """
    link = WEBSITE_URL + tag.h3.a['href']
    name = tag.h3.a.string
    try:
        poster = tag.find('img')['loadlate']
    except:
        poster = None
    try:
        genres = tag.p.find('span', class_='genre').string.strip()
    except AttributeError:
        genres = None
    try:
        runtime = tag.p.find('span', class_='runtime').string
    except AttributeError:
        runtime = None
    try:
        certificate = tag.p.find('span', class_='certificate').string
    except AttributeError:
        certificate = None
    try:
        rating = tag.find('strong').string
    except AttributeError:
        rating = None
    try:
        year = tag.find('span', class_="lister-item-year text-muted unbold").string[-5:-1]
    except TypeError:
        year = None
    try:
        votes = tag.find('p', class_='sort-num_votes-visible').findAll('span')[1].string
    except:
        votes = 0
    try:
        director = tag.find('div', class_='lister-item-content').findAll('p')[2].find('a').string
    except AttributeError:
        director = None
    return [name, genres, link, runtime, certificate, rating, year, votes, director, poster]


def save_to_database(movie_df):
    """
    Save data to database.

    Parameters
    ----------
    movie_df : DataFrame
    """
    print("Saving data to database...")
    df = movie_df.copy()
    conn = sqlite3.connect(DATABASE_NAME)

    df.director.fillna("", inplace=True)
    dd = dict(zip(df.director.unique(), range(len(df.director.unique()))))
    df['directorId'] = [dd[i] for i in df.director]

    movie = df[['name', 'genres', 'link', 'runtime', 'certificate', 'rating', 'year', 'votes', 'directorId', 'poster']]
    movie.to_sql("Movies", conn, index=False, if_exists='replace')

    director = df[['directorId', 'director']].drop_duplicates()
    director.columns = ["Id", "director"]
    director.to_sql("Director", conn, index=False, if_exists='replace')
    conn.close()
    print("Data has been successfully saved.")


def read_from_database():
    """
    Read data from database.

    Returns
    -------
    DataFrame
        Information of all movies.
    """
    with sqlite3.connect(DATABASE_NAME) as con:
        movies = pd.read_sql("SELECT * FROM movies JOIN director ON movies.directorId=director.Id", con=con)
        movies.rating = np.float32(movies.rating)
        movies.year = np.int32(movies.year)
        votes = np.int32([vote.replace(",", "") for vote in movies.votes])
        movies.votes = votes
    return movies


def get_data_from_website():
    """

    Scrape websites for movie data, and save to database.

    """
    CACHE_DICT = load_cache()
    # All the links
    links = [DATA_SOURCE + f"&start={i}&ref_=adv_nxt" for i in range(51, TOTAL_MOVIES, 50)]
    links.insert(0, DATA_SOURCE)

    total_df = pd.DataFrame()
    count = 0
    for link in links:
        count += 1
        print(count)
        single_page = make_url_request_using_cache(link, CACHE_DICT)
        single_page = fetch_data_of_single_page(single_page)
        temp_df = pd.DataFrame(data=single_page, columns=INFORMATION_LIST)
        total_df = pd.concat([total_df, temp_df]).reset_index(drop=True)
    save_to_database(total_df)
    print(total_df.head())


def read_data():
    """
    Read data from database if there exist one. Otherwise, scrape websites.

    Returns
    -------
    movies_df : DataFrame
        All information we needed for a single movie.
    """
    if os.path.exists(DATABASE_NAME):
        movies_df = read_from_database()
    else:
        get_data_from_website()
        movies_df = read_from_database()
    return movies_df


class Filter:
    """
    The filter object can select movies satisfied all conditions users give.
    """
    def __init__(self, data_df):
        self.data = data_df

    def choose_by_rating(self, start=0, end=10):
        return (end >= self.data.rating) & (self.data.rating >= start)

    def choose_by_year(self, start=1915, end=np.inf):
        return (end >= self.data.year) & (self.data.year >= start)

    def choose_by_votes(self, start=1000, end=np.inf):
        return (end >= self.data.votes) & (self.data.votes >= start)

    def choose_by_genres(self, genre_list=['Drama']):
        if len(genre_list) == 0:
            return self.data.year == self.data.year
        temp = self.data.genres.apply(lambda x: genre_list[0] in x)
        if len(genre_list) > 1:
            for genre in genre_list[1:]:
                temp = np.logical_or(temp, self.data.genres.apply(lambda x: genre in x))
        return temp

    def choose_by_genres_opposite(self, genre_list=['Drama']):
        if len(genre_list) == 0:
            return self.data.year == self.data.year
        temp = self.data.genres.apply(lambda x: genre_list[0] in x)
        if len(genre_list) > 1:
            for genre in genre_list[1:]:
                temp = np.logical_or(temp, self.data.genres.apply(lambda x: genre in x))
        return np.logical_not(temp)


movies = read_data()
f1 = Filter(movies)
app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('user.html')


@app.route("/", methods=['POST'])
def test():
    rating_min = np.float(request.form.get('min'))
    rating_max = np.float(request.form.get('max'))
    try:
        votes_min = np.float(request.form.get('votes_min'))
    except:
        votes_min = 1000
    try:
        votes_max = np.float(request.form.get('votes_max'))
    except:
        votes_max = np.inf
    try:
        year_min = np.float(request.form.get('year_min'))
    except:
        year_min = 1900
    try:
        year_max = np.float(request.form.get('year_max'))
    except:
        year_max = np.inf

    checked_genres = []
    for genre in GENRES:
        if request.form.get(genre):
            checked_genres.append(genre)

    dislike_genres = []
    for genre in GENRES:
        if request.form.get(genre+"1"):
            dislike_genres.append(genre)

    mask1 = f1.choose_by_rating(rating_min, rating_max)
    mask2 = f1.choose_by_votes(votes_min, votes_max)
    mask_temp = np.logical_and(mask1, mask2)
    mask3 = f1.choose_by_year(year_min, year_max)
    mask_temp = np.logical_and(mask_temp, mask3)
    mask4 = f1.choose_by_genres(checked_genres)
    mask_temp = np.logical_and(mask_temp, mask4)
    mask5 = f1.choose_by_genres_opposite(dislike_genres)
    mask_temp = np.logical_and(mask_temp, mask5)
    subset = movies[mask_temp].reset_index(drop=True)

    subset['poster_html'] = [f'<img src="{i}">' for i in subset.poster]

    # Choose columns
    all_columns = ['poster_html', 'name', 'genres', 'link', 'runtime', 'certificate', 'rating', 'year', 'votes',
                   'director']
    columns = ['Poster', 'Name', "Genres", "movie_link", "Runtime", "Certificate", "Rating", "Year", "Votes", "Director"]
    choice = []
    for i, column in enumerate(columns):
        if request.form.get(column):
            choice.append(all_columns[i])

    if len(choice) == 0:
        return subset[all_columns].to_html(escape=False, render_links=True)
    else:
        return subset[choice].to_html(escape=False, render_links=True)


app.run(debug=True)

