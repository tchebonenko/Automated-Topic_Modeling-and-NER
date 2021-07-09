# Fourthbrain + GLG Capstone Deeplearning Project
Named Entity recognition, Hierarchical topic modeling
![image](https://user-images.githubusercontent.com/7217067/115755705-3f0f7700-a363-11eb-86a2-588f802f7fa9.png)


# 1.    Background and Significance of Project


GLG powers great decisions through our network of experts. GLG receives hundreds of requests a day from clients seeking insights on topics ranging from the airline industry’s ability to cope with COVID-19 to the zebra mussel infestations in North America. GLG’s goal is to match each request to a topic specialist in their database. This project is a Natural Language Processing (NLP) challenge aimed at improving the topic/keyword detection process from the client submitted reports and identifying the underlying patterns in submitted requests over time.  The primary challenges include Named Entity Recognition (NER) and Pattern Recognition for Hierarchical Clustering of Topics.

Typically, the client requests we receive comprise a form with unstructured free text with screening questions. Thus, we have a need to group these requests into common topics – to better understand and service demand. This project is aimed to increase the resourcefulness of the current data pipelines for efficient data storage and retrieval.

GitHub: https://github.com/Milan-Chicago/GLG-Automated-Meta-data-Tagging/tree/main (switch to branch ‘drm’ for NER code)

# 2.    Related Work (Papers, github)


Topic modeling:
David M. Blei, Andrew Y. Ng, Michael I. Jordan , Latent Dirichlet Allocation , Journal of Machine Learning Research 3 (2003) 993-1022

Link:
https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil, Universal Sentence Encoder, arXiv:1803.11175, 2018.

Link: https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
	

Named Entity Recognition:

spaCy NER (for baselining): https://spacy.io/models/en. Precision/recall of baseline en_core_web_sm is ~ 0.86
Finetuning BERT for NER (next level): https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/, https://github.com/chambliss/Multilingual_NER. Fine tuning a BERT model on the NER task gives a validation accuracy of ~0.95 after just one epoch of fine tuning. We plan to add code for computing precision/recall on individual named entities. 
A next step is to try DistilBERT, which is a lighter weight BERT model (and hence probably faster in deployment): https://arxiv.org/abs/1910.01108. Another lightweight transformer model is MobileBERT: https://arxiv.org/abs/2004.02984. 

# 3.    Explanation of Data sets

There will be two kinds of data sets for this project:

https://components.one/datasets/all-the-news-2-news-articles-dataset/
2.7 million news articles and essays from 27 American publications. Includes date, title, publication, article text, publication name, year, month, and URL (for some). Articles mostly span from 2013 to early 2020.

https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set. This particular dataset has 47959 sentences and 35178 unique words. 

# 4.    Explanation of Processes (Methods)

Main model chosen for the topic modeling is Latent Dirichlet Allocation (LDA). This model is a form of unsupervised learning that views documents as bags of words (i.e. order does not matter). LDA works by first making a key assumption: the way a document was generated was by picking a set of topics and then for each topic picking a set of words.

Without a deep dive into mathematical theory behind the model, we can consider LDA as a probabilistic distribution of topics in documents and distribution of words in each topic.

List of topics for a document is defined from a per-document distribution over topics in the training set of documents and ordered by the topic probability given words of this document. 
For example, an output of LDA model for a document might be
65% “sports” + 25%”news” + 3%”politics” +2%”war” + 1%”economy”  + …

where “sports”, ”news”, “politics”, “war” and “economy” are topic names.

A topic name is defined as the most frequent key-word in the documents that have this topic as the most probable topic in the LDA output for the documents.

To increase LDA model efficiency and accuracy we applied the following steps:
Dimensionality reduction: each noun phrase and verb in the text is replaced by its keyphrase/verb that is most semantically similar (using BERT based transformer)
Texts are transformed to collections of key-phrases and verbs
LDA models are trained on the transformed texts

Advantage of LDA: 
defines the distribution of topics in a new document based on words of the document that were seen during model training
provides probability of a topic that can be used to select leading topic of a document or if it should be classified as miscellaneous since it does not fit topics covered before
can be easily scaled to big databases and updated with new texts

Hierarchical topic modeling approach:





Topic modeling algorithm defines First, Second and Third level topics for the submitted text as follows:
Each level of topics is the output of that level topic modeling algorithm trained on the related subset of texts (First level – full training sample, Second level – sample of texts that have the same first level topic, Third level – sample of texts that have the same first and second level topics)
Not more than 10 topics are selected for first level, 5 for second level and 3 for third (ideally number of topics should be defined through testing which is highly costly from time and resources perspective)
A text is assigned a topic with the highest probability if it is > 20% or “Miscellaneous” on any level of topic modeling

Model candidates:
LDA + Bag-of-Words using only nouns, verbs, adjectives, and adverbs converted to their initial form (lemmas) - base line
LDA + key-phrases that are labels of semantically similar noun phrases and verbs. 
Process: 
Each noun phrase and verb in the texts is  transformed to embedding vector using Universal Sentence Encoder (transformer based on BERT)

Noun phrase is a subject/object in a sentence with its modifier. For Example:
”conference panel”, “staged car theft”, “fixed chronic airflow obstruction”, “best institution”. 

Embedding vectors from (a) are grouped/clustered if they have cosine similarity > 70%
A word/phrase in a cluster is considered as it’s label (key-phrase) if it has highest frequency in the collection of texts among all members of the cluster
		
Results:
Dictionary dimensionality is reduced from 419327 to 179257.
Average cluster size is 18 phrases (median 10 phrases with less than 10% 
of clusters with only 1 phrase/word)

Each text in the training sample is converted to a collection of key-phrases by replacing its noun phrases and verbs with keyword/phrases and deleting other words
LDA and LDA predictions are performed on the transformed texts
Each unseen text is first transformed to a collection of noun-phrases and verbs which are replaced by a corresponding similarity cluster’s label. After that LDA models make their predictions where models are chosen hierarchically: first level defines second level models and they both defining third level model

Data for model training

After data exploration and cleaning only major news publications are selected that have more than 10 sections and less than 100 news sections. Sections with potentially too broad variation of topics or words used are deleted from the data.

Selected news sections:

Number of observations in the data selected:

To mimic the length of GLG client’s requests, only first 10 sentences were selected from each news article.

First level LDA model is trained on 33982 news articles from 'CNN', 'Economist', 'Gizmodo', and  'Wired' publications.

Subsequent levels are trained on a subset of news that have the same first (first and second) level topics.

NER

For NER, one can use spaCy pretrained pretty much right out of the box as a baseline, and it is relatively easy to feed data through to obtain named entities. However, since this is a pretrained model, there is little flexibility in terms of types of NER datasets that one can train/finetune on. As a next step, we fine tune a BERT model (using the HuggingFace library built on PyTorch) on a publicly available NER dataset (https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus).

Due to time constraints, we could not do the following:
Try to fine tune a transformer on a more specialised NER dataset, such as one pertaining to healthcare (such as this covid specific one https://xuanwang91.github.io/2020-03-20-cord19-ner/)
Try to fine tune a lighter weight transformer model such as distilbert and/or mobilebert, and compare performance. We plan to incorporate some level of flexibility into the finetuning pipeline.

With regards to the second bullet point above, using a distilbert or mobilebert model will be more suited to large scale production models, due to their smaller size (and hence quicker latency).

# 5.    Explanation of Outcomes 

Hierarchical topic modeling:

Although the LDA model provides insights on what keywords from each topic, choosing the best suited topic name can only be done manually. 

We used Tf-Idf metric for noun phrases that appear in the documents within each topic to define most frequent and relevant noun phrases. This noun phrase is chosen to be the name of a topic.

The output of LDA topic naming on the train set of news looks as following:


where 
each column corresponds to an article
“publication” and “section” rows display initial values for each article in the train set
last three rows represent derived hierarchical topic names

As can be seen, although  many topic names seem to be reasonable, not every topic has a sensible name that falls in the publication section theme.

NER

After trying out the spaCy baseline, we decided to go one step further with a BERT based model. Since the arrival of BERT, transformer based architectures have shown great versatility in use across many language based tasks (NER, next sentence prediction, masked token prediction, machine translation, to name a few). (More recently transformer based models have been shown to be valuable for vision tasks as well, however we will not discuss that here.) 

While the BERT model is overall an accurate tagger, transformers struggle with long range dependencies (and generally long input corpora) -- this is an active area of research and investigation. However, for this use case, we decided to train a more accurate tagger at the cost of input length. The primary reason we chose the BERT based architecture is for its flexibility discussed above -- both in model selection (choosing between different transformer architectures), as well as dataset customization (curating/choosing a custom dataset to fine tune on). The idea of named entity recognition is to be able to identify named entities -- e.g. people, places, and organizations. This is an important aspect of information retrieval because it enables extraction of core types of information from unstructured text.

Below is an example of an input and output of our named entity model, served with fastAPI. The underlying model is a BERT base with a linear layer on top, fine-tuned on the Kaggle NER dataset referenced above for three epochs.

Input

'{
  "text": "Divy Murli was born in India, grew up in Boise, Idaho and was educated at UCSB and Stanford."
}'



Output

{
  "predictions": {
    "extracted_entities": "[('Divy', 'B-per'), ('Murli', 'I-per'), ('India', 'B-geo'), ('Boise', 'B-geo'), ('Idaho', 'B-geo'), ('UCSB', 'B-org'), ('Stanford', 'B-org')]"
  }
}

Token level precision-recall for the model we trained for NER are outlined below.

While pretrained spaCy models work great out of the box, what is nice about using transformers is (a) their ease in customization and (b) their ease in evaluation. We see above that our model performs best on location-based tokens, and comparatively worse on organization-based tokens.

# 6.    System Design and Ethical Considerations

Given we are using public news and healthcare data sources, the intended outcome is a summary or LDA modeling of the proposed text.  Future use cases could potentially involve use of more PII or Medical data use cases.
Need to be aware of cultural biases that arise by country.   
We did not modify our strategy based on these assumptions.   
Currently we do not anticipate any issues that can potentially arise.   
# Automated-Topic_Modeling-and-NER
