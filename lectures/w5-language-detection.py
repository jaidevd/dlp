#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
from tqdm import tqdm


# In[2]:


en = load_dataset('facebook/voxpopuli', 'en', streaming=True, split='train').take(100)
es = load_dataset('facebook/voxpopuli', 'es', streaming=True, split='train').take(100)
ro = load_dataset('facebook/voxpopuli', 'ro', streaming=True, split='train').take(100)


# In[3]:


data = []
for x, y, z in tqdm(zip(en, es, ro)):
    data.extend([x, y, z])


# In[5]:


from datasets import Dataset


# In[8]:


def mygen():
    for i in data:
        yield i

ds = Dataset.from_generator(mygen)


# In[9]:


ds


# In[10]:


ds.save_to_disk('language-identification')


# In[11]:


from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
wav_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')


# In[ ]:


def process_wav(sample):
    enc = processor(sample['audio']['array'], sampling_rate=16_000, padding=True, return_tensors='pt')
    out = wav_model(enc).last_hidden_state.squeeze().mean(axis=0)
    return out

encodings = []
for sample in tqdm(ds):
    encodings.append(process_wav(sample))


# In[19]:


encodings


# In[23]:


L = encodings.map(lambda x: {'size': len(x['input_values'][0])}, remove_columns=encodings.column_names)


# In[35]:


sample = encodings[0]['input_values']
type(sample)


# In[36]:


processor(ds[0]['audio']['array'], sampling_rate=16_000, return_tensor='pt')


# In[38]:


x = processor(ds[0]['audio']['array'], sampling_rate=16_000, return_tensors='pt')['input_values']


# In[43]:


wav_model(x).last_hidden_state.squeeze().mean(axis=0)


# In[ ]:




