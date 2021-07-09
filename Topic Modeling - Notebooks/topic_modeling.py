import re
# topic modeling libraries
from gensim import models, corpora

# supporting libraries
import pickle
import pandas as pd
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import spacy
nlp = spacy.load("en_core_web_md")


import torch
from transformers import (
    BertForTokenClassification,
    BertTokenizer
)


################  Topics for new texts using pretrained model ####################


def get_LDA_model(path):
    lda_model = models.LdaMulticore.load(path)
    return lda_model


def get_LDA_dictionary(path):
    with open(path, 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
        dictionary = pickle.load(f)
    return dictionary


def get_list_of_sentences(text):
    spacy_doc = nlp(text)
    list_of_sents = list(spacy_doc.sents)[:10]
    list_of_sents = [s.text for s in list_of_sents]
    return list_of_sents


def get_list_of_lemmas(text):
    # extract 'NOUN', 'VERB', 'ADJ', 'ADV' from text
    # if they are not stop-words, have length>2 and have only alphabetic characters
    selected_POSs = ['NOUN', 'VERB', 'ADJ', 'ADV']

    spacy_doc = nlp(text)
    list_of_lemmas = [word.text.lower() for word in spacy_doc if (word.is_stop == False) &
                      (len(word.text) > 2) &
                      (word.is_alpha) &
                      (word.pos_ in selected_POSs)]
    return list_of_lemmas


def get_top_topic_index(text,
                        params={"LDA_dictionary_path": "./output/lda_keywords/dictionary1.pickle",
                                "LDA_model_path": "./output/lda_keywords/LDA_model1"
                                }
                        ):

    if "lda_keywords" in params['LDA_model_path']:
        list_of_words = text.split(" ")
    else:
        list_of_words = get_list_of_lemmas(text)

    # load topic dictionary
    with open(params['LDA_dictionary_path'], 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        dictionary = pickle.load(f)

    doc2bow = dictionary.doc2bow(list_of_words)

    # load topic model
    lda = get_LDA_model(params['LDA_model_path'])

    # get (topic,proba) tuples
    vector = lda[doc2bow]

    topic_number, proba = sorted(vector, key=lambda item: item[1])[-1]

    if proba < 0.2:
        return -1, -1
    else:
        return topic_number, proba

################  Train LDA model ####################


def prepare_for_modeling(data_path, model_type="LDA-KeyWords",
                         params={"TEXT_prepared_df": pd.DataFrame({}),
                                 "save_LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                                 "words_column": "all_key_words"
                                 },
                         verbose=1):
    if model_type == "LDA-KeyWords":
        """
        params={"TEXT_prepared_df": pd.DataFrame({}),
                 "save_LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                 "words_column": "all_key_words"
                }
        """

        if len(params['TEXT_prepared_df']) > 0:
            # load data for LDA
            df_data = params['TEXT_prepared_df']
            if verbose == 2:
                print("loaded data shape:", df_data.shape)
        else:
            if verbose == 2:
                print("No data is provided")
            return False

        words_column = params['words_column']
        df_data[words_column] = df_data[words_column].apply(lambda x: [w.replace(' ', '_') for w in x
                                                                       if len(w) > 1
                                                                       ])
        # get all unique key_words
        tmp_list = df_data[words_column].tolist()
        set_of_words = set([w for sublist in tmp_list for w in sublist])

        if verbose == 2:
            print('\nNumber of unique key-words for topic modeling dictionary:',
                  len(set_of_words))

        # delete empty lists of words
        df_data = df_data[df_data[words_column].apply(len) > 0]

        # create a vocabulary for the LDA model
        dictionary = corpora.Dictionary(df_data[words_column])

        # save dictionary
        with open(params["save_LDA_dictionary_path"], 'wb') as f:
            # Pickle the LDA dictionary using the highest protocol available.
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        if verbose == 2:
            print("LDA dictionary file is saved to:",
                  params["save_LDA_dictionary_path"])

            print('\nNumber of texts processed: ', dictionary.num_docs)
            print('Number of extracted key-words: ', len(dictionary.token2id))
            print('\nEach text is represented by list of ', len(dictionary.token2id),
                  " tuples: \n\t\t(key-words's index in bag-of-words dictionary, key-words's term frequency)")

        # count the number of occurrences of each distinct token in each document
        df_data['doc2bow'] = df_data['all_key_words'].apply(
            lambda x: dictionary.doc2bow(x))

    if model_type == "LDA":
        """
        params={"TEXT_prepared_df": pd.DataFrame({}),
                                 "save_LDA_dictionary_path": "./output/lda/dictionary.pickle",
                                 "text_column": "text"
                                 }
        """
        if len(params['TEXT_prepared_df']) > 0:
            # load data for LDA
            df_data = params['TEXT_prepared_df']
            print("loaded data shape:", df_data.shape)
        elif len(data_path) > 0:
            print("Preparing data for LDA...")
            df_data = pd.read_csv(params['data_path'])
            df_data['list_of_lemmas'] = df_data[words_column].apply(
                lambda text: get_list_of_lemmas(text))
            print("Data for LDA shape:", df_data.shape)
        else:
            return False

        # get all unique lemmas
        tmp_list = df_data['list_of_lemmas'].apply(set).apply(list).tolist()
        list_of_words = [w for sublist in tmp_list for w in sublist]

        # count words' document frequencies in the corpus
        w_freq_counter = collections.Counter(list_of_words)
        s_w_freq = pd.Series(w_freq_counter)
        if verbose == 2:
            print('\nTotal number of unique Lemmas: ', len(s_w_freq))
            print("\nDistribution of lemmas' document counts: ")
            print(pd.DataFrame(s_w_freq.describe(percentiles=[
                  0.55, 0.65, 0.75, 0.85, 0.95, 0.97, 0.99])).T)

        # select upper and lower boundary for lemmas' count
        up_pct = s_w_freq.quantile(0.99)
        low_pct = 3  # s_w_freq.quantile(0.50)
        if verbose == 2:
            print("\nDeleting too frequent and too rare words...")
            print('Lemma count upper bound:', up_pct)
            print('Lemma count lower bound:', low_pct)

        # select Lemmas
        selected_words = set(s_w_freq[(s_w_freq > low_pct)
                                      & (s_w_freq <= up_pct)].index)
        if verbose == 2:
            print('\nList of words for topic modeling dictionary is reduced from',
                  len(s_w_freq), 'to', len(selected_words))

        # select words in each article if they belong to chosen list of words
        df_data['selected_words'] = df_data['list_of_lemmas'].apply(lambda x:
                                                                    [l for l in x if l in selected_words])
        # delete empty lists of words
        df_data = df_data[df_data['selected_words'].apply(len) > 0]

        # create a vocabulary for the LDA model
        dictionary = corpora.Dictionary(df_data['selected_words'])

        # save dictionary
        with open(params["save_LDA_dictionary_path"], 'wb') as f:
            # Pickle the LDA dictionary using the highest protocol available.
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        if verbose == 2:
            print("LDA dictionary file is saved to:",
                  params["save_LDA_dictionary_path"])

            print('\nNumber of texts processed: ', dictionary.num_docs)
            print('Number of extracted lemmas: ', len(dictionary.token2id))
            print('\nEach text is represented by list of ', len(dictionary.token2id),
                  " tuples: \n\t\t(lemma's index in bag-of-words dictionary, lemma's term frequency)")

        # count the number of occurrences of each distinct token in each document
        df_data['doc2bow'] = df_data['selected_words'].apply(
            lambda x: dictionary.doc2bow(x))

    return df_data


def train_model(model_type="LDA-KeyWords",
                params={"num_topics": 10,
                        "LDA_prepared_df": pd.DataFrame({}),
                        "LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                        "save_LDA_model_path": "./output/lda_keywords/LDA_model"
                        },
                verbose=1):
    if model_type == "LDA-KeyWords":
        """
        params={"num_topics": 10,
                "LDA_prepared_df": pd.DataFrame({}),
                "LDA_dictionary_path": "./output/lda_keywords/dictionary.pickle",
                "save_LDA_model_path": "./output/lda_keywords/LDA_model"
                }
        """
        print("Training LDA with semantically similar clusteres ow words (NOUN_PHRASEs and VERBs)")
    if model_type == "LDA":
        """
        params={"num_topics": 10,
                "LDA_prepared_df": pd.DataFrame({}),
                "LDA_dictionary_path": "./output/lda/dictionary.pickle",
                "save_LDA_model_path": "./output/lda/LDA_model"
                }
        """
        print("Training LDA with only lemmas of NOUNs, VERBs, ADJs and ADVs")

    if len(params['LDA_prepared_df']) > 0:
        # load data for LDA
        df_data = params['LDA_prepared_df']
        if verbose == 2:
            print("loaded data shape:", df_data.shape)
    else:
        return False

    # download LDA dictionary
    dictionary = get_LDA_dictionary(params['LDA_dictionary_path'])

    # create document-term matrix for LDA
    if verbose == 2:
        print("\nCreating document-term matrix for LDA...")
    doc_term_matrix = list(df_data['doc2bow'].values)

    # define the model with chosen number of topics
    num_topics = params['num_topics']
    if verbose == 2:
        print("\nTraining LDA model with ", num_topics, " topics...")

    LDA = models.LdaMulticore
    result_lda_model = LDA(corpus=doc_term_matrix,
                           num_topics=num_topics,
                           id2word=dictionary,
                           passes=20,
                           chunksize=4000,
                           random_state=3)
    # Save model to disk
    result_lda_model.save(params["save_LDA_model_path"])
    print("LDA model file is saved to:", params["save_LDA_model_path"])

    # get topics
    df_data['infered_topics'] = df_data['doc2bow'].apply(lambda d:
                                                         sorted(result_lda_model[d],
                                                                key=lambda x: x[1],
                                                                reverse=True))
    # select top index
    df_data['top_topic'] = df_data['infered_topics'].apply(
        lambda x: x[0][0] if x[0][1] >= 0.2 else -1)
    df_data['top_topic_proba'] = df_data['infered_topics'].apply(
        lambda x: x[0][1])

    if verbose == 2:
        print(
            'Top topic indexes are selected. NOTE "-1" corresponds to top topic with probability < 20%')
    return df_data

################  Name extracted topics ####################


def name_topic(df, words_column, topic_words):
    # print(df.shape)
    words_to_count = list(df[words_column])
    words_to_count = [w for l in words_to_count for w in l if len(w) > 1]
    #print(len(words_to_count), topic_words)
    words_to_count = [w for w in words_to_count if w in topic_words]
    if len(words_to_count) > 0:
        try:
            words_to_count = [w.replace("_", " ").lower()
                              for w in words_to_count]
            words_to_count = [w[0].upper() + w[1:] for w in words_to_count]
            # print(words_to_count[:5])

            c = collections.Counter(words_to_count)
            return c.most_common(3)[0][0]
        except:
            print(c)
    else:
        return "ERROR"


def get_topic_names(df_result, topic_column, words_column,
                    LDA_model_path, num_topics, num_words=20):
    list_dfs = []
    all_topics = list(set(df_result[topic_column]))
    topic_words_dict = get_dict_topic_words(
        LDA_model_path, num_topics, num_words=20)

    try:
        for topic in all_topics:
            #print (topic)
            df_topic = df_result[df_result[topic_column] == topic].copy()
            #print(topic, df_topic.shape)
            topic_words = topic_words_dict[int(str(topic)[-1])]
            df_topic[topic_column +
                     "_name"] = name_topic(df_topic, words_column, topic_words)
            list_dfs.append(df_topic)

        df_res = pd.concat(list_dfs)

        return df_res[topic_column + "_name"]
    except:
        print(topic_words_dict)


def get_dict_topic_words(LDA_model_path, num_topics, num_words=20):
    # get top num_words words that appear both in topic  and in text list_of_words
    lda = get_LDA_model(LDA_model_path)
    x = lda.show_topics(num_topics=num_topics,
                        num_words=num_words, formatted=False)
    topic_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

    return dict(topic_words)
################  Process unseen text ####################


def clean_NPs(np):
    # clean noun phrases from stop-words
    tmp_no_stop_words = [w for w in np if w.is_stop == False]

    # make only last word as lemma
    if len(tmp_no_stop_words) > 0:
        tmp_lemmas = [w.text for w in tmp_no_stop_words[:-1]] + \
            [tmp_no_stop_words[-1].lemma_]
    else:
        tmp_lemmas = []

    tmp_atleast_one_alpha = [
        w for w in tmp_lemmas if len(re.sub(r"\d|\W", "", w)) > 0]
    tmp_result = [w.lower() for w in tmp_atleast_one_alpha if len(w) > 0]

    return " ".join(tmp_result)


def get_NPs_Vs(text):
    doc = nlp(text)

    # extract noun phrases
    NPs = [np for np in doc.noun_chunks]

    # delete stop-words ("the", "a", "your" etc.) and clean NPs
    NPs = [clean_NPs(np) for np in NPs]
    NPs = [np for np in NPs if len(np) > 1]

    # extract verd lemmas
    Vs = [word.text for word in doc if (word.is_stop == False) &
          (len(word.text) > 2) &
          (word.is_alpha) &
          (word.pos_ in ['VERB'])]
    return NPs + Vs


def get_embeddings(list_of_words):
    embeddings = [nlp(w).vector for w in list_of_words]
    return np.appay(embeddings)


def get_word_embeddings(df_data, column="word"):
    df_tmp = df_data
    df_tmp['emb_vector'] = df_tmp[column].apply(lambda w: nlp(w).vector)
    w_vectors = np.array(list(df_tmp['emb_vector']))

    columns = ["emb_" + str(i) for i in range(300)]
    df_data[columns] = w_vectors
    del df_data['emb_vector']

    return df_data


def get_keyword(row, df_emb):
    take_cluster_name = row['take_cluster_name']
    sim_max_index = row['sim_max_index']

    if take_cluster_name:
        return df_emb['cluster_label'].iloc[sim_max_index]
    else:
        return ""


def predict_topics(text,
                   params={"topics_df_path": './output/lda_keywords/topics.pickle',
                           "word_embeddings": './output/lda_keywords/word_embeddings.pickle',
                           "first_dictionary_path": "./output/lda/dictionary1.pickle",
                           "first_LDA_model_path": "./output/lda/LDA_model1"
                           }
                   ):

    if 'lda_keywords' in params["topics_df_path"]:
        # get clustered words' embedings and cluster names(key words) from train corpus
        with open(params["word_embeddings"], 'rb') as f:
            df_emb = pickle.load(f)

        # extract keywords from text
        NPs_and_Vs = get_NPs_Vs(text)
        df_text_words = pd.DataFrame(NPs_and_Vs, columns=['text_words'])
        df_text_emb = get_word_embeddings(
            df_text_words, column="text_words")

        # find closest word in train corpus and get cluster name
        columns = ["emb_" + str(i) for i in range(300)]
        sim_values = cosine_similarity(df_text_emb[columns], df_emb[columns])
        max_sim_values = np.max(sim_values, axis=1)
        df_text_words['take_cluster_name'] = max_sim_values >= 0.7
        df_text_words['sim_max_index'] = np.argmax(sim_values, axis=1)
        df_text_words['keyword'] = df_text_words.apply(
            get_keyword, axis=1, args=[df_emb])

        words_for_LDA = list(df_text_words['keyword'])
        words_for_LDA = [w for w in words_for_LDA if len(w) > 0]

        text = " ".join(words_for_LDA)

    # load pre-trained topics
    LDA_topics_df_path = params["topics_df_path"]
    with open(LDA_topics_df_path, 'rb') as f:
        df_topics = pickle.load(f)
    # df_topics.head(1)

    # first level
    first_LDA_dict_path = params["first_dictionary_path"]
    first_LDA_model_path = params["first_LDA_model_path"]
    t1, t1_proba = get_top_topic_index(text,
                                       params={"LDA_dictionary_path": first_LDA_dict_path,
                                               "LDA_model_path": first_LDA_model_path
                                               }
                                       )

    # second level
    second_LDA_dict_path = first_LDA_dict_path[:-
                                               7] + "_" + str(t1 + 1) + ".pickle"
    second_LDA_model_path = first_LDA_model_path + "_" + str(t1 + 1)
    t2, t2_proba = get_top_topic_index(text,
                                       params={"LDA_dictionary_path": second_LDA_dict_path,
                                               "LDA_model_path": second_LDA_model_path
                                               }
                                       )

    # third level
    third_LDA_dict_path = first_LDA_dict_path[:-7] + \
        "_" + str(t1 + 1) + "_" + str(t2 + 1) + ".pickle"
    third_LDA_model_path = first_LDA_model_path + \
        "_" + str(t1 + 1) + "_" + str(t2 + 1)
    t3, t3_proba = get_top_topic_index(text,
                                       params={"LDA_dictionary_path": third_LDA_dict_path,
                                               "LDA_model_path": third_LDA_model_path
                                               }
                                       )

    # get topic names
    if t1 == -1:
        t1_name = "misc."
    else:
        t1_name = df_topics[df_topics['first_level_topic']
                            == t1]['first_level_topic_name'].iloc[0]

    if t2 == -1:
        t2_name = "misc."
    else:
        t2_name = df_topics[df_topics['second_level_topic'] == str(t1) +
                            '.' + str(t2)]['second_level_topic_name'].iloc[0]
    if t3 == -1:
        t3_name = "misc."
    else:
        t3_name = df_topics[df_topics['third_level_topic'] == str(t1) +
                            '.' + str(t2) + '.' + str(t3)]['third_level_topic_name'].iloc[0]

    dict_output = {'first_level_topic': t1,
                   'first_level_topic_name': t1_name,
                   'first_level_topic_proba': t1_proba,
                   'second_level_topic': t2,
                   'second_level_topic_name': t2_name,
                   'second_level_topic_proba': t2_proba,
                   'third_level_topic': t3,
                   'third_level_topic_name': t3_name,
                   'third_level_topic_proba': t3_proba
                   }
    return dict_output

################  NER using pretrained model ####################


class BERT_NER_inference(object):
    """
    This class is meant to load a pretrained BERT NER model from a saved
    checkpoint onto the CPU to be used for inference
    """

    def __init__(self, model_path):
        self.device = torch.device("cpu")  # load model onto CPU for inference
        self.tokenizer = self.get_tokenizer()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=18,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.tag_values = checkpoint["tag_values"]

    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case=False)
        return tokenizer

    def predict(self, text_input):
        self.model.eval()
        tokenized_input = self.tokenizer.encode(text_input)
        input_ids = torch.tensor([tokenized_input])
        with torch.no_grad():
            output = self.model.forward(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)
        # just extract the named entities:
        # TODO: probably a more efficient way to do this but this is enough for now
        extracted_named_entites = []
        for entity in zip(new_tokens, new_labels):
            if entity[1] != "O":
                extracted_named_entites.append(entity)
        return {
            "extracted_entities": repr(extracted_named_entites)
        }


def get_entity_name(ent_list, entity="org"):
    selected_ent = [e for e in ent_list if entity in e[1]]

    ent_names = []
    current_name = ''
    for e in selected_ent:
        if e[1] == "B-" + entity:
            ent_names.append(current_name)
            current_name = e[0]
        else:
            current_name = current_name + " " + e[0]

    ent_names.append(current_name)
    ent_names = [n for n in ent_names if len(n) > 0]

    return ent_names


def get_all_named_entities(text, model_path):
    inference = BERT_NER_inference(model_path)

    # separate text into sentenses
    list_sentences = get_list_of_sentences(text)

    # get all entities from the text sentence by sentense
    all_entities = []
    for sent in list_sentences:
        inf = inference.predict(sent)['extracted_entities']
        ent_list = inf[3:-3].split("'), ('")
        ent_list = [s.split("', '") for s in ent_list]
        all_entities.extend(ent_list)

    all_entities = [e for e in all_entities if len(e[0]) > 0]

    # extract only PER, GEO and ORG, entities
    ent_dict = {"Names": [],
                "Places": [],
                "Organisations": []
                }
    for key in ent_dict:
        if key == "Names":
            entity = "per"
        if key == "Places":
            entity = "geo"
        if key == "Organisations":
            entity = "org"
        if len(all_entities) > 0:
            tmp_list = get_entity_name(all_entities, entity=entity)
            tmp_list = [e for e in tmp_list if "[SEP]" not in e]
            ent_dict[key].extend(tmp_list)

    for key in ["Names", "Places", "Organisations"]:
        ent_dict[key] = list(set(ent_dict[key]))
    return ent_dict
