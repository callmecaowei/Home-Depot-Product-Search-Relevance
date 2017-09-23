import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
#读入数据集
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_desc = pd.read_csv('../input/product_descriptions.csv')
#融合数据集
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
#文本预处理，看关键字是否被包含
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])
#看关键字出现了多少次
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

import Levenshtein
df_all['dist_in_title'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)
#tfidf
df_all['all_texts']=df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)
print(dictionary)
class MyCorpus(object):
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

# 这里这么折腾一下，仅仅是为了内存friendly。面对大量corpus数据时，直接存成一个list，会使得整个运行变得很慢。

corpus = MyCorpus()
#有了语料库，就可以计算tfidf
from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)
#计算两个橘子的相似度
from gensim.similarities import MatrixSimilarity

# 先把刚刚那句话包装成一个方法
def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

# 然后，我们创造一个cosine similarity的比较方法
def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    # 本来sim输出是一个array，我们不需要一个array来表示，
    # 所以我们直接cast成一个float
    return float(sim[0])
#计算两个句子相似度
df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
import nltk
#word2vec
# ##nltk也是自带一个强大的句子分割器。
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer.tokenize(df_all['all_texts'].values[0])
sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
sentences = [y for x in sentences for y in x]
from nltk.tokenize import word_tokenize
w2v_corpus = [word_tokenize(x) for x in sentences]
from gensim.models.word2vec import Word2Vec

model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)
#计算一个句子的平均的word2vec，先拿到全部的vocabulary
vocab = model.vocab

# 得到任意text的vector
def get_vector(text):
    # 建立一个全是0的array
    res =np.zeros([128])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model[word]
            count += 1
    return res/count
#计算两个句子的相似度
from scipy import spatial
# 这里，我们再玩儿个新的方法，用scipy的spatial

def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)
df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']),axis=1)
#丢掉文本文件
df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)

df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
test_ids = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
rf = RandomForestRegressor(n_estimators=30, max_depth=6)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('submission.csv',index=False)
# model1=xgb.train(X_train)
# #y_pred1 = model1.predict(X_test)
#pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('submission.csv',index=False)
