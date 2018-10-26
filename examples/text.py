
# coding: utf-8

# In[1]:


from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality


# # Text example

# An example of creating a language model and then transfering to a classifier.

# In[2]:


path = untar_data(URLs.IMDB)
path


# Open and view the independent and dependent variables:

# In[3]:


df = pd.read_csv(path/'train.csv', header=None)
df.head()


# In[4]:


classes = read_classes(path/'classes.txt')
classes[0], classes[1]


# Create a `DataBunch` for each of the language model and the classifier:

# In[5]:


data_lm = TextLMDataBunch.from_csv(path)
data_clas = TextClasDataBunch.from_csv(path, vocab=data_lm.train_ds.vocab)


# [fast.ai](http://www.fast.ai/) has a pre-trained English model available that we can download.

# In[6]:


datasets.download_wt103_model()


# We'll fine-tune the language model:

# In[7]:


learn = RNNLearner.language_model(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'])
learn.unfreeze()
learn.fit(2, slice(1e-4,1e-2))


# Save our language model's encoder:

# In[8]:


learn.save_encoder('enc')


# Fine tune it to create a classifier:

# In[9]:


learn = RNNLearner.classifier(data_clas)
learn.load_encoder('enc')
learn.fit(3, 1e-3)

