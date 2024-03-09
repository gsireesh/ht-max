import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.width', None)



def find_index_span(string, substring):

    start_index = string.find(substring)
    end_index = start_index + len(substring) - 1

    return start_index, end_index

def split_column_entity(string):

    t = string.split('\t')
    entity_value = t[-1]
    g = t[0].split(' ')
    entity_label = g[0]
    span_0, span_1 = g[-2], g[-1]

    return entity_label, int(span_0), int(span_1), entity_value

def split_column_relation (string):

    s = string.split(" ")
    relation_type = s[0]
    Arg_1 = s[1].split(":")[-1]
    Arg_2 = s[2].split(":")[-1]

    return relation_type, Arg_1, Arg_2


def get_entity_relation_by_keyword(ann_path,text_path,keywords):


    ann_df = pd.read_csv(ann_path, sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)

    '''
    Entity Data Processing
    '''

    entity_df = ann_df[ann_df[1].str.startswith('T')]
    relation_df = ann_df[ann_df[1].str.startswith('R')]

    en_columns = ['entity_index', "entity_label"]

    entity_df.columns = en_columns

    entity_df['entity_span_0'] = entity_df["entity_label"].apply(lambda x: split_column_entity(x)[1])
    entity_df['entity_span_1'] = entity_df["entity_label"].apply(lambda x: split_column_entity(x)[2])
    entity_df['entity_value'] = entity_df["entity_label"].apply(lambda x: split_column_entity(x)[3])
    entity_df['entity_label'] = entity_df["entity_label"].apply(lambda x: split_column_entity(x)[0])



    with open(text_path, 'r') as file:

        article_text = file.read()

    Sentences = article_text.split(".")

    entity_df['entity_span_0'] = entity_df['entity_span_0'].astype(int)
    entity_df['entity_span_1'] = entity_df['entity_span_1'].astype(int)


    for i, sentence in enumerate(Sentences):

        span_start, span_end = find_index_span(article_text, sentence)

        for index in range(entity_df.shape[0]):

            if entity_df.loc[index,"entity_span_0"] >= span_start and entity_df.loc[index,"entity_span_1"] <= span_end:

                entity_df.loc[index, "sentence"] = sentence


    for word in keywords:

        useful_sentences = entity_df.loc[entity_df['entity_value'].str.contains(word), 'sentence'].tolist()

        useful_entity_df = entity_df[entity_df['sentence'].isin(useful_sentences)]


    useful_entity_index = useful_entity_df['entity_index'].tolist()


    '''
    Relation Data Processing
    '''

    re_columns = ['relation_index', "relation_label"]

    relation_df.columns = re_columns



    relation_df['relation_argument_0'] = relation_df["relation_label"].apply(lambda x: split_column_relation(x)[1])

    relation_df['relation_argument_1'] = relation_df["relation_label"].apply(lambda x: split_column_relation(x)[2])

    relation_df['relation_label'] = relation_df["relation_label"].apply(lambda x: split_column_relation(x)[0])



    useful_relation_df = relation_df[relation_df['relation_argument_0'].isin(useful_entity_index) | relation_df['relation_argument_1'].isin(useful_entity_index)]


    print(useful_entity_df)
    print(useful_relation_df)

    return useful_entity_df, useful_relation_df



if __name__ == "__main__":

    ann_path = "test.ann"
    text_path = "test.txt"
    keywords = ["creep"]

    useful_entity_df, useful_relation_df = get_entity_relation_by_keyword(ann_path=ann_path,text_path=text_path,keywords=keywords)





