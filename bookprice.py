import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

data = pd.read_csv("E:/ML PROJECTS/books_data.csv")
print(data.head())

data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')

# Check the data types again and display histogram
print(data.info())

fig = px.histogram(data.dropna(), x='average_rating',
                   nbins=30,
                   title='Distribution of Average Ratings')
fig.update_xaxes(title_text='AVERAGE RATING')
fig.update_yaxes(title_text='FREQUENCY')
fig.show()

'''top_authors=data['authors'].value_counts().head(10)
top_authors.columns = ['Author', 'Number of Books']

fig=px.bar(top_authors, x=top_authors.values,y=top_authors.index,orientation='h',
           labels={'x':'NUMBER OF BOOKS','y':'Author'},
           title='NUMBER OF BOOKS PER AAUTHOR')
fig.show()'''
top_authors = data['authors'].value_counts().head(10).reset_index()
top_authors.columns = ['Author', 'Number of Books']

fig = px.bar(top_authors, x='Number of Books', y='Author', orientation='h',
             labels={'Number of Books': 'NUMBER OF BOOKS', 'Author': 'Author'},
             title='NUMBER OF BOOKS PER AUTHOR')

fig.show()


