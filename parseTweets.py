#!/usr/bin/env python

import os, re, json, codecs

# sourcedir = '/Users/mark/Research/2018 ELO/data03'
sourcedir = '../data03'
#targetdir = '/Users/mark/Research/2018 ELO/processed_texts_Spacy03'
targetdir = 'texts'

for fname in os.listdir(sourcedir):
	if fname.endswith('json'):
		print(fname)
		nfname = re.sub('json$', 'txt', fname)
#		text = open(os.path.join(sourcedir, fname)).read().decode('utf-8')
		with codecs.open(os.path.join(sourcedir, fname), 'r', 'utf-8') as f:
			tweets = json.load(f, encoding='utf-8')
		text = u''
		for tweet in tweets:
#			t = tweet['text'] # for twitterscraper
			t = tweet['tweet'] # for Twint
			t = re.sub('http\S*', '', t)
			t = re.sub('pic.twitter\S*', '', t)
			t = re.sub('[\:\s]+$', '.', t)
			t = re.sub(ur'(\w)$', r'\1\.', t)
#			print(t)
#			if not re.match('\W$', t): t = t + '.'
			text = text + '\n\n' + t
#		parsed = tagger.tag_text(text)
		text = re.sub(ur'([\.\?\!])([^\.\?\!])', r'\1 \2', text)
		file = open(os.path.join(targetdir, nfname), 'wb')
		file.write(text.encode('utf8'))
		file.close()
