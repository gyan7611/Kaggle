
#Importing the libraries

import pandas as pd
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import sys

def Predict_Sarcasm():

	#Loading train and test sets

	train_path,test_path,out_path = sys.argv[1:]

	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)


	#Merging the train and test sets for feature creation
	X_columns = [x for x in df_test.columns]
	X_train = df_train[X_columns]
	X_test = df_test

	merged_df = X_train.append(X_test)


	#Initialising the sentiment modul
	sent = SentimentIntensityAnalyzer()
	tokenizer = RegexpTokenizer(r'\w+')
	# In[9]:

	merged_df['neutral score']=merged_df['tweet'].apply(lambda x: sent.polarity_scores(x)['neu'])
	merged_df['pos score']=merged_df['tweet'].apply(lambda x: sent.polarity_scores(x)['pos'])
	merged_df['neg score']=merged_df['tweet'].apply(lambda x: sent.polarity_scores(x)['neg'])
	merged_df['length of tweet']=merged_df['tweet'].apply(lambda x: len(x))
	merged_df['Capital count']=merged_df['tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))
	merged_df['Quote count']=merged_df['tweet'].apply(lambda x: x.count("'"))


	#Positive and Negative Sentiment Scores were working fine but what worked better was the different between the score
	#Creating the column for difference between pos and neg scores
	merged_df['difference score']=merged_df['pos score']-merged_df['neg score']



	#Creating a list of tweets (DOCUMENTS)    
	doc_set=[i for i in merged_df['tweet'] if type(i)==str] 


	#Cleaning the text for POS related features
	final=[]
	for doc in doc_set:
	    raw_desc = doc.lower()
	    tokens = tokenizer.tokenize(raw_desc) 
	    pos_tag=nltk.pos_tag(tokens)
	    tok=[y for (x,y) in pos_tag if len(x)>1 and ("RB" in y or "JJ" in y or 'NN' in y                                                 or 'VB' in y or 'IN' in y)]
	    final.append(tok)


	counter_dict=[Counter(x) for x in final]

	#Creating the counts of different POS tags as a feature
	#Nouns,Adjectives,Verbs,Adverbs and Interjections are the tags that were targetted

	adj,adv,verb,noun,inter=[],[],[],[],[]

	for x in counter_dict:
	    if "RB" in x:
	        adv.append(x['RB'])
	    else:
	        adv.append(0)
	    if "JJ" in x:
	        adj.append(x['JJ'])
	    else:
	        adj.append(0)
	    if "NN" in x:
	        noun.append(x['NN'])
	    else:
	        noun.append(0)
	    if "VB" in x:
	        verb.append(x['VB'])
	    else:
	        verb.append(0)
	    if 'IN' in x:
	        inter.append(x['IN'])
	    else:
	        inter.append(0)
	merged_df['ADV_Count'] = adv
	merged_df['ADJ_Count'] = adj   
	merged_df['VERB_Count']=verb
	merged_df['NOUN_Count']=noun
	merged_df['Inter_Count']=inter


	#Counts were fine but what appeared more interesting was the fraction of these tags in the tweet.
	#Creating features for what % of the tweet are the different POS tags
	adj_share=[]
	adv_share=[]
	verb_share=[]
	noun_share=[]
	inter_share=[]
	for i,row in merged_df.iterrows():
	    adj_share.append(row['ADJ_Count']/float(len(row['tweet'].split())))
	    adv_share.append(row['ADV_Count']/float(len(row['tweet'].split())))
	    verb_share.append(row['VERB_Count']/float(len(row['tweet'].split())))
	    noun_share.append(row['NOUN_Count']/float(len(row['tweet'].split())))
	    inter_share.append(row['Inter_Count']/float(len(row['tweet'].split())))

	merged_df['% of adj']=adj_share
	merged_df['% of adv']=adv_share
	merged_df['% of verb']=verb_share
	merged_df['% of noun']=noun_share
	merged_df['% of inter']=inter_share


	train = merged_df[:len(df_train)]
	test = merged_df[len(df_train):]

	train['label'] = df_train['label']


	#Features were one thing but from some consistent text mining and using some word clouds ,we saw something interesting
	#Turns out people are very consistent in using some tags like #sarcasm ,#not when they are tweeting something sarcastic
	#So we formed a rule,whenever a tweet contains #sarcasm ,#sarcastic ,#SARCASM ,#not ,#serious ,#seriouly, we label it as sarcastic

	rule_based_assignment=[]

	for index,row in test.iterrows():
	    
	    if '#sar' in row['tweet'].lower() or '#not' in row['tweet'].lower() or 'serious' in row['tweet'].lower():
	            
	        rule_based_assignment.append('sarcastic')
	    else:
	        rule_based_assignment.append('non-sarcastic')        


	#Finally, the features were ready to be trained.
	#Played around with a number of algorithms , Random Forest , Naive Bayes , SVM
	#Finally used XGBoost,and the parameters were decided using 5-fold CV score

	#Importing the requirements
	import xgboost
	xgc=xgboost.XGBClassifier(learning_rate=0.001,n_estimators=25)


	#Creating a list of Xs of the model
	X_model = ['neutral score','length of tweet','Capital count','Quote count','difference score', \
		u'% of adj',u'% of adv', u'% of verb', u'% of noun','% of inter']


	#Fitting the model
	xgc.fit(train[X_model],train['label'])

	#Making the predictions
	xg_pred=xgc.predict(test[X_model])



	#Using rule based results and combining them with the model results
	final_preds=['sarcastic' if 'sarcastic' in x else 'non-sarcastic' for x in zip(xg_pred,rule_based_assignment)]



	#Creating the final dataframe by
	test['label']=final_preds
	test[['ID','label']].to_csv(out_path,index=False)

if __name__=='__main__':
	Predict_Sarcasm()






