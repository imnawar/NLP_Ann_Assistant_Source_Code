import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from bot import telegram_chatbot
import re
import numpy 
import tensorflow 
import random 
import json
import tflearn
import pickle

with open("intents.json") as file:
	data = json.load(file)

try:
	with open("data.pickle", "rb") as f:
		words, labels, traning, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	traning = [] 
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		traning.append(bag)
		output.append(output_row)


	traning = numpy.array(traning)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, traning, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(traning[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model.fit(traning, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]


	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)	


def sen_analysis(s):
	result = SentimentIntensityAnalyzer().polarity_scores(s)
	negative = result['neg']
	positive = result['pos']
	print("input : pos:neg")
	print(positive)
	print(negative)
	f = re.match("^((?![Nn]o).)*$",s)
	f2 = re.match("^((?![Dd]ont).)*$",s)
	if (negative > positive):
		print("first if ")
		return False
	elif(negative == positive):
		if(re.match("^((?![Nn]o).)*$",s)):
			print("No match")
			if(re.match("^((?![Dd]ont).)*$",s)):
				return True
			else:
				return False
	else:
		print("else ")
		return True

bot = telegram_chatbot("config.cfg")

def chat():
	
	isymptoms ={"fever":0 , "drycough":0, "tiredness":0, "sorethroat":0 , "diarrhoea":0, "lossoftasteorsmell":0, "achesandpains":0 , "headache":0, "conjunctivitis":0}
	user_probability = {}
	sym_us ={}
	us_factor = {}
	update_id = None 
	while True:
		updates = bot.get_updates(offset=update_id)
		updates = updates["result"]
		factor = 1
		if updates:
			for item in updates:
				update_id = item["update_id"]
				try:
					from_ = item["message"]["from"]["id"]
					if from_ not in user_probability:
						user_probability[from_] = 0 
						sym_us[from_] = isymptoms.copy()
						us_factor[from_]=1
					inp = item["message"]["text"]
					results = model.predict([bag_of_words(inp, words)])
					results_index = numpy.argmax(results)
					tag = labels[results_index]
					for tg in data["intents"]:
						if tg['tag'] == tag:
							t = tag
							responses = tg['responses']
							if(tg['tag'] == "days"):
								inpt= nltk.word_tokenize(inp.lower())
								if "yes" in inpt:
									us_factor[from_]=1
								else:
									us_factor[from_]=0.5
							if((t in isymptoms) and sen_analysis(inp) and sym_us[from_][t]==0):
								sym_us[from_][t]=tg['probability']

					if (tag == "end"):
						for x in isymptoms:
							user_probability[from_] = user_probability[from_]+ float(sym_us[from_][x])
						user_probability[from_]= user_probability[from_]*us_factor[from_]
						user_probability[from_] = format(user_probability[from_], '.2f')
						rep = random.choice(responses) + str(user_probability[from_]) +"%"
					elif(tag == "goodbye"):
						del sym_us[from_] 
						del user_probability[from_]
						del us_factor[from_]
						rep = random.choice(responses)
					else:
						rep = random.choice(responses)
				except:
					message = None
					rep = None
				bot.send_message(rep, from_)
				
chat()


