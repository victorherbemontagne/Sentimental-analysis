import sys
import numpy as np
import os
os.getcwd()

sys.path.append("C:\git\eki-python_library")


# La bibli que  l'on utilise pour faire l'analyse est le Doc2Vec de gensim
import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# Nltk va nous donner sa méthode pour tokenizer les mots et les detokenizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()

# librairie utilisée pour scrapper les infos Facebook
from ekimetrics.api.facebook import *

import logging

from sklearn.cross_validation import cross_val_score

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

def compute_score(clf,X, y):
    xval = cross_val_score(clf, X, y, cv = 5)
    return(np.mean(xval))

# Utilisé pour la partie où l'on va chercher à trier le language

from langdetect import detect,detect_langs,DetectorFactory 
DetectorFactory.seed = 0
sid = SentimentIntensityAnalyzer()

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression

class LabeledLineSentence(object):

	def __init__(self, sources):
		self.sources = sources
		self.sentences = self.to_array()
		flipped = {}

		# make sure that keys are unique
		#for key, value in sources.items():
		#	if value not in flipped:
		#		flipped[value] = [key]
		#	else:
		#		raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

	def to_array(self):
		sentences = []
		for prefix, source in self.sources.items():
			for item_no, line in enumerate(source):
				sentences.append(LabeledSentence(
					line, [prefix + '_%s' % item_no]))
		return sentences

	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences


class FacebookCommentsLabeling(object):


	def __init__(self,file_paths, data = ""): # You can either give a Json of Data or pass directly pass a Dataframe of comments
		if data == "":
			self.comments = []
			self.page_data = []
			for file_path in file_paths:
				page = Page(file_path = file_path)
				self.page_data.append(page)
				comments_page = page.get_all_comments()
				for comment in comments_page:
					self.comments.append(comment.message)
		else: 
			self.comments = data
		print("/n Doc of size ",len(self.comments)," well imported")



	def prepare_data(self):
		matchs = []
		rests = []
		print(' Begin the match classification..')
		for page in self.page_data:
			(match,rest) = page.get_all_matchReactionComment()
			matchs.append(match) #liste de dictionnaires
			rests.append(rest) # same
		match = matchs.pop(0)
		for match_page in matchs:
			for key, items in match_page.items():
				if key in match.keys():
					match[key]+=items
				else:
					match[key] = items
		rests_util = [comment for rest in rests for comment in rest]
		print('Match classificatin finished..beggin the nltk analysis..')
		dictScore = {'comments':[],'compound':[],'neg':[],'neu':[],'pos':[],'textblob_score':[],'textblob_subjectivity':[]}
		sid = SentimentIntensityAnalyzer() # qui vas nous aider à bâtir notre première approche senti
		print('Exploring the rest data set of size ',len(rests_util))
		for comment in rests_util:
			ss = sid.polarity_scores(comment.message)
			dictScore['comments'].append(comment)
			for k in sorted(ss):
				dictScore[k].append(ss[k])
			txtblob_score = TextBlob(comment.message)
			dictScore['textblob_score'].append(txtblob_score.sentiment[0])
			dictScore['textblob_subjectivity'].append(txtblob_score.sentiment[1])
		self.rest_analyzed = pd.DataFrame().from_dict(dictScore)
		self.rest_analyzed['tags'] = 0
		print('Finish analyzed the rest dataset..')
			##############
		#Maintenant on va traiter les comments qui ont matché
		# On donne une note pleine par rapport à la réaction à laquelle ils sont associés
		dictReaction={'comments':[],'compound':[],'neg':[],'neu':[],'pos':[],'textblob_score':[],'textblob_subjectivity':[], 'tags': []}
		##Ici on crée les DF avec les comments suivant dont est sur qu'ils sont positif vs ceux dont on est moins sur
		print('Begin to explore the match')
		for reaction in match.keys():
			if reaction == "ANGRY" or reaction=="SAD":
				for comment in match[reaction]:
					dictReaction['comments'].append(comment)
					dictReaction['tags'].append(-1)

					ss = sid.polarity_scores(comment.message)
					for k in sorted(ss):
						dictReaction[k].append(ss[k])
					txtblob_score = TextBlob(comment.message)
					dictReaction['textblob_score'].append(txtblob_score.sentiment[0])
					dictReaction['textblob_subjectivity'].append(txtblob_score.sentiment[1])

			if reaction == "WOW" or reaction == "LOVE" or reaction == "HAHA":
				for comment in match[reaction]:
					dictReaction['comments'].append(comment)
					dictReaction['tags'].append(1)

					ss = sid.polarity_scores(comment.message)
					for k in sorted(ss):
						dictReaction[k].append(ss[k])
					txtblob_score = TextBlob(comment.message)
					dictReaction['textblob_score'].append(txtblob_score.sentiment[0])
					dictReaction['textblob_subjectivity'].append(txtblob_score.sentiment[1])

				if reaction == "LIKE":
					for comment in match[reaction]:
						dictReaction['comments'].append(comment)
						dictReaction['tags'].append(0)

						ss = sid.polarity_scores(comment.message)
						for k in sorted(ss):
							dictReaction[k].append(ss[k])
					txtblob_score = TextBlob(comment.message)
					dictReaction['textblob_score'].append(txtblob_score.sentiment[0])
					dictReaction['textblob_subjectivity'].append(txtblob_score.sentiment[1])
		print('Finish explore match data set..')
		commentsTaged = pd.concat([pd.DataFrame().from_dict(dictReaction),self.rest_analyzed],ignore_index = True) # les comments tagués

		#Maintenant on va prendre les deux differentes bases et on va les rassembler dans une seule grande base

		# On va pouvoir utiliser ce set qualifié pour entrainer un modèle et le tester sur le reste de ma base
		# On va peut être essayer de la booster un peu plus en rajoutant des comments dont on est sur en se basant sur NLTK

		certain_comments = commentsTaged.query('tags != 0')
		incertain_comments = commentsTaged.query('tags == 0')
		incertain_comments['tags'] = incertain_comments['pos'].apply(lambda x: 1 if x >= 0.5 else 0)

		incertain_comments = commentsTaged.query('tags == 0')

		incertain_comments['tags'] = incertain_comments['compound'].apply(lambda x: 1 if x >= 0.6 else 0)

		certain_comments = pd.concat([
									certain_comments, 
									incertain_comments[incertain_comments.tags == 1]
									],
									ignore_index=True)



		incertain_comments = incertain_comments[incertain_comments.tags != 1]

		comments_neg = certain_comments.query('tags == -1')
		comments_pos = certain_comments.query('tags == 1')
			# on transforme en dictionnaire

		comments_neg = comments_neg.to_dict(orient = 'list') #{'comments':[comment1,comment2],'tags':[tag1,tag2,..],...}
		comments_pos = comments_pos.to_dict(orient = 'list') #{'comments':[comment1,comment2],'tags':[tag1,tag2,..],...}

			############################################################################################################

		# Je dois d'abord transformer comments_neg en liste de la forme [[comment1,tag1],[comment2,tag2],...]

		def parsing_dict_for_gensim(comments):
			comments_util = []
			for index_comment in range(len(comments['comments'])-1):
				comments_util.append([comments['comments'][index_comment].message,
									comments['tags'][index_comment]
									])
			return(comments_util)

		# comment_neg_util de la forme : [[comment1,tag1],[comment4,tag4],[comment2,tag2],[comment3,tag3]]
		comment_neg_util = parsing_dict_for_gensim(comments_neg)
		comment_pos_util = parsing_dict_for_gensim(comments_pos)


		comment_neg_trie = trie_langue(comment_neg_util)
		comment_pos_trie = trie_langue(comment_pos_util)

		self.comment_neg_trie = [nltk.word_tokenize(comment[0]) 
							for comment in comment_neg_trie['long_comment']['en']+comment_neg_trie['long_comment']['fr']
						]
		self.comment_pos_trie = [nltk.word_tokenize(comment[0]) 
							for comment in comment_pos_trie['long_comment']['en']+comment_pos_trie['long_comment']['fr']
						]
		self.rest_analyzed_tokenized = [nltk.word_tokenize(comment.message) 
							for comment in list(self.rest_analyzed['comments'])]

		print('Data well prepared, ready to train')

	def launch_analyzis(self,choosen_model = 'LinearSVC',nbr_epochs = 20,save_model =True, return_prediction = False):

		self.model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=2)
		length_neg_set = len(self.comment_neg_trie)

		self.sources = {
					'train_pos':self.comment_pos_trie[:length_neg_set//2],
				  	'train_neg':self.comment_neg_trie[:length_neg_set//2],
				  	'test_neg':self.comment_neg_trie[length_neg_set//2:],
				  	'test_pos':self.comment_pos_trie[length_neg_set//2:],
				  	'to_analyse' : self.rest_analyzed_tokenized #on doit lui rentrer une liste
				}
		sentences = LabeledLineSentence(self.sources)
		build_vocab_data = sentences.to_array()
		self.model.build_vocab(build_vocab_data)
		for epoch in range(nbr_epochs):
			logger.info('Epoch %d' % epoch)
			self.model.train(sentences.sentences_perm(),
					total_examples= len(build_vocab_data), 
					epochs = self.model.iter)

		if save_model:
			self.model.save(choosen_model,'_gensim_last')

		#The  vocabulary is build, we can then take the vector that represent each comment of the training set in order to train our classifier
		train_arrays = np.zeros((length_neg_set, 100))
		train_labels = np.zeros(length_neg_set)
		for i in range(length_neg_set//2):
			prefix_train_pos = 'train_pos_' + str(i)
			prefix_train_neg = 'train_neg_' + str(i)
			train_arrays[i] = self.model.docvecs[prefix_train_pos]
			train_arrays[length_neg_set//2 + i] = self.model.docvecs[prefix_train_neg]
			train_labels[i] = 1
			train_labels[length_neg_set//2 + i] = 0


		#We can then take the vector that represent each comment of the testing set in order to test our classifier
		test_arrays = np.zeros((length_neg_set-2, 100))
		test_labels = np.zeros(length_neg_set-2)
		for i in range(length_neg_set//2-2):

		    prefix_test_pos = 'test_pos_' + str(i)
		    prefix_test_neg = 'test_neg_' + str(i)
		    test_arrays[i] = self.model.docvecs[prefix_test_pos]
		    test_arrays[length_neg_set//2 + i] = self.model.docvecs[prefix_test_neg]
		    test_labels[i] = 1
		    test_labels[length_neg_set//2 + i] = 0

		
		# Creating the TFIDF matrice


		if choosen_model == 'LinearSVC':
			from sklearn.svm import LinearSVC
			self.classifier = LinearSVC()
			self.classifier.fit(train_arrays,train_labels)
			score = compute_score(self.classifier,test_arrays,test_labels)
			print(choosen_model,' model successfully trained with cross_val_score : ',score)
		if choosen_model == 'RandomForest':
			from sklearn.ensemble import RandomForestClassifier
			self.classifier = RandomForestClassifier()
			self.classifier.fit(train_arrays,train_labels)
			score = compute_score(self.classifier,test_arrays,test_labels)
			print(choosen_model,' model successfully trained with cross_val_score : ',score)
		if return_prediction:
			print('Beginning prediction for unlabelled data..')

			#For the prediction first we can predict the results with our model, than
			data = list(self.rest_analyzed['comments'])
			size_rest = len(data)
			to_predict = []
			for i in range(size_rest):
				prefix_analysis = 'to_analyse_' + str(i)
				to_predict.append(self.model.docvecs[prefix_analysis])
			prediction_array = self.classifier.predict(to_predict)
			OUT = []
			for index in range(len(prediction_array)):
				OUT.append([data[index],prediction_array[index]])
		return(OUT)


def trie_langue(corpus):
	format_OUT = {'small_comment': [],
				  'long_comment': {'en': [], 'fr': []}
				  }
	
	def delete_name(tokenized_comment):
		words_to_del = []
		for index_word in range(len(tokenized_comment)-1):
			if (tokenized_comment[index_word].lower() != tokenized_comment[index_word] and 
				tokenized_comment[index_word+1].lower() != tokenized_comment[index_word+1]):

				words_to_del.append(tokenized_comment[index_word])

				words_to_del.append(tokenized_comment[index_word+1])

		for word_to_del in words_to_del:
			if word_to_del in tokenized_comment:
				index_word_to_del = tokenized_comment.index(word_to_del)
				try:
					tokenized_comment.pop(index_word_to_del)
				except IndexError:
					pass
		return(tokenized_comment)
	size = len(corpus)
	tokenized_corpus = [{'comment':comment, 'tokenized_comment':nltk.word_tokenize(comment[0])} for comment in corpus]
	
	for comment in tokenized_corpus: # comment de la forme : [{'comment':[comment, tag],'tokenized_comment':[word,word,..]}]
		if len(comment['tokenized_comment']) < 5:
			if comment['comment'][0] != '':
				format_OUT['small_comment'].append(comment['comment'])
		else:
			try:
				lang = detect(comment['comment'][0])
				if lang == 'en' or lang == 'fr':
					format_OUT['long_comment'][lang].append(comment['comment'])
			except:
				print('Can\'t find language')
				pass

	small_comment_util = [] #liste de [[comment,tag],...,[commentN,tagN]]
	long_comment_util = {'en': [], 'fr': []}
	
	for comment in format_OUT['small_comment']: # comment de la forme  [comment, tag]
		print(comment)
		if delete_name(nltk.word_tokenize(comment[0])) != []:
			small_comment_util.append([detokenizer.detokenize(delete_name(nltk.word_tokenize(comment[0])),return_str=True),
									   comment[1]
									  ])
	for comment in format_OUT['long_comment']['en']: # même format
		long_comment_util['en'].append([detokenizer.detokenize(delete_name(nltk.word_tokenize(comment[0])),return_str=True),
									   comment[1]
									  ])
	for comment in format_OUT['long_comment']['fr']: # même format
		long_comment_util['fr'].append([detokenizer.detokenize(delete_name(nltk.word_tokenize(comment[0])),return_str=True),
									   comment[1]
									   ])
	
	OUT = {
		'small_comment':small_comment_util,
		'long_comment':long_comment_util,
		}
	return(OUT)
