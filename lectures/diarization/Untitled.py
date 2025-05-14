#!/usr/bin/env python
# coding: utf-8

# In[1]:


from faster_whisper import WhisperModel


# In[2]:


whisper = WhisperModel('base', device='cuda', compute_type='int8')


# In[56]:


file = "TEST-1.wav"
segments, info = whisper.transcribe(file, language='en', task='transcribe', beam_size=5, best_of=5)


# In[54]:


len([k for k in segments])


# In[7]:


import pandas as pd


# In[57]:


df = pd.DataFrame([
    (seg.start, seg.end, seg.text) for seg in segments
], columns=['start', 'end', 'text'])
df.head()


# In[58]:


df.tail()


# In[9]:


import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

speaker_embedding = PretrainedSpeakerEmbedding('speechbrain/spkrec-epaca-voxceleb', device=torch.device('cuda'))


# In[11]:


from pyannote.audio import Audio
from pyannote.core import Segment


# In[14]:


crop_segments = [Segment(r.start, min(r.end, 120)) for _, r in df.iterrows()]
crop_segments[:3]


# In[15]:


audio = Audio()


# In[18]:


crops = [audio.crop(file, seg)[0] for seg in crop_segments]


# In[20]:


crops[0].shape


# In[21]:


crops[1].shape


# In[25]:


x = crops[0]


# In[27]:


x.shape


# In[31]:


embeddings = [speaker_embedding(x.unsqueeze(0)).squeeze(0) for x in crops]


# In[32]:


import numpy as np


# In[33]:


embeddings = np.r_[embeddings]
embeddings.shape


# In[36]:


from sklearn.cluster import AgglomerativeClustering


# In[37]:


clus = AgglomerativeClustering(n_clusters=2)
clus.fit(embeddings)

clus.labels_


# In[38]:


df['speaker'] = [f"Speaker {i}" for i in clus.labels_]


# In[39]:


df.head()


# In[42]:


df[['start', 'end']] = df[['start', 'end']].astype(int)


# In[47]:


def seconds_to_srt_time(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = 0  # As we're only given seconds, we'll set milliseconds to zero
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Updated function to convert DataFrame with integer times to SRT format
def dataframe_to_srt(df):
    srt_content = ''
    for i, row in df.iterrows():
        start_time = seconds_to_srt_time(row['start'])
        end_time = seconds_to_srt_time(row['end'])
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{row['speaker']}\n\n"
    return srt_content

srt = dataframe_to_srt(df)
with open('TEST-1.srt', 'w') as fout:
    fout.write(srt)


# In[51]:


pd.options.display.max_colwidth = 100
df[['text', 'speaker']]


# In[59]:


import faster_whisper


# In[60]:


faster_whisper.__version__


# In[63]:


len(whisper.supported_languages)


# In[64]:


from sklearn.datasets import load_iris


# In[68]:


X, y = load_iris(return_X_y=True, as_frame=True)


# In[66]:


from sklearn.cluster import KMeans


# In[76]:


iris = load_iris()
labels = iris.target_names


# In[93]:


km = KMeans(n_clusters=3, random_state=42)
km.fit(X)

pd.DataFrame(np.c_[labels, km.labels_]).drop_duplicates().sort_values(0)


# In[79]:


labels = labels[y]


# In[ ]:




