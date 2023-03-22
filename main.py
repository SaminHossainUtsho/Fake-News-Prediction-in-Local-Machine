import nltk
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.linear_model import LogisticRegression
from tkinter import *
from functools import partial
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def stemming(content):
	nltk.download('stopwords')
	port_stem = PorterStemmer()
	stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
	stemmed_content = stemmed_content.lower()
	stemmed_content = stemmed_content.split()
	stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
	stemmed_content = ' '.join(stemmed_content)
	return stemmed_content


def label(l):
	if l == 1:
		return "Fake News"
	else:
		return "Real News"


def test_model(title, author):
	train_dataset = pd.read_csv('train.csv')

	train_dataset = train_dataset.fillna('')

	train_dataset['Combined_column'] = train_dataset['author'] + ' ' + train_dataset['title']  # +' '+ train_dataset['text']

	X = train_dataset.drop(columns='label', axis=1)
	Y = train_dataset['label']

	author_title = author + ' ' + title  # +' '+ 'text'
	test_news = {"text": [author_title]}
	test_news = pd.DataFrame(test_news)
	test_news["text"] = test_news["text"].apply(stemming)
	stemmed_content = test_news["text"]


	X = train_dataset['Combined_column'].values
	Y = train_dataset['label'].values

	vectorizer = TfidfVectorizer()
	vectorizer.fit(X)
	Z = vectorizer.transform(X)

	new_test_vec = vectorizer.transform(stemmed_content)
	# print(new_test_vec)
	filename = 'Log_Reg_model_default.sav'

	model = pickle.load(open(filename, 'rb'))
	predict = model.predict(new_test_vec[:,:17128])
	print("Prediction : ", label(predict))

	str  =  label(predict)
	class Window(Frame):
		def __init__(self, master=None):
			Frame.__init__(self, master)
			self.master = master
			self.pack(fill=BOTH, expand=1)

			text = Label(self, text="Prediction : "+str)
			text.place(x=100, y=90)

	# text.pack()

	root = Tk()
	app = Window(root)
	root.wm_title("Tkinter window")
	root.geometry("300x300")
	root.mainloop()
	return



def check_it(title, author):
	print("Title entered :", title.get())
	print("Author entered :", author.get())
	t = title.get()
	a = author.get()

	test_model(t,a)
	return

#window
tkWindow = Tk()
tkWindow.geometry('800x500')
tkWindow.title('Fake News Prediction')

#username label and text entry box
idLabel = Label(tkWindow, text="Enter ID : ",width="30", height="5").grid(row=20, column=5)
id = StringVar()
idEntry = Entry(tkWindow, textvariable=id).grid(row=20, column=10)

titleLabel = Label(tkWindow, text="Enter Title : ",width="30").grid(row=40, column=5)
title = StringVar()

titleEntry = Entry(tkWindow, textvariable=title).grid(row=40, column=10)

authorLabel = Label(tkWindow, text="Enter Author : ",width="30", height="5").grid(row=60, column=5)
author = StringVar()

authorEntry = Entry(tkWindow, textvariable=author).grid(row=60, column=10)

textLabel = Label(tkWindow, text="Enter text : ",width="30",height="5").grid(row=80, column=5)
text = StringVar()
textEntry = Entry(tkWindow, textvariable=text).grid(row=80, column=10)


check_it = partial(check_it, title, author)

checkButton = Button(tkWindow, text="Check", command=check_it).grid(row=100, column=10)

tkWindow.mainloop()