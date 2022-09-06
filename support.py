import csv
import time
import copy

import numpy as np
import pdfplumber
import pandas as pd
import fitz
import io
from PIL import Image
import os, shutil
from fpdf import FPDF
import enchant
import pprint
from pluralizer import Pluralizer
from pdfreader import SimplePDFViewer, PageDoesNotExist
from pattern.text.en import singularize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words as words_dataset_from_nltk
from wordfreq import word_frequency
from tabulate import tabulate
import codecs
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def replace_if_in_replacement_table(some_word,keys_to_be_replaced,replacement_table):
    if (some_word in keys_to_be_replaced):
        return replacement_table[some_word]
    else:
        return some_word

def filter_of_trash_characters(word,disallowed_characters,english_alphabet):
    copy_of_word_ = copy.copy(word)
    need_to_future_process_ = True
    # если не символ в строке не входит в алфавит, то удалить символ
    pure_row_= ""
    for simbol in copy_of_word_:
        if(simbol not in english_alphabet):
            continue
        else:
            pure_row_+=simbol
    # pure_row_ = copy_of_word_.translate({ord(c): None for c in disallowed_characters})

    # если осталась просто одна буква, то удалить в дальнейшем
    if pure_row_ in english_alphabet:
        need_to_future_process_ =False
    # если осталась просто пустая строка, то удалить в дальнейшем
    if pure_row_=="":
        need_to_future_process_ =False
    return need_to_future_process_, pure_row_

def filter_word_list(word_list,filter,disallowed_characters,english_alphabet):
    output_list = []
    for word in word_list:
        need_to_future_process_, pure_row_ = filter(word,
                                                    disallowed_characters=disallowed_characters,
                                                    english_alphabet=english_alphabet)
        if need_to_future_process_:
            output_list.append(pure_row_)
    return  output_list


def filter_for_tagged_words(word,parts_of_speech):
    need_to_future_process_ = False
    if word in parts_of_speech:
        need_to_future_process_ =True
    return need_to_future_process_,word

def filter_tagged_words(word_tuples,filter,parts_of_speech):
    output_list= []
    for word_tuple in word_tuples:
        need_to_future_process_, pure_row_ = filter(word_tuple[1],parts_of_speech)
        if need_to_future_process_:
            output_list.append(word_tuple)
    return output_list

def refresh_all_word_dict(all_word_dict):
    output = {}
    for key,value in all_word_dict.items():
        if len(value)!=0:
            output.update({key:list(set(value))})
    return output

def get_num_of_element_in_dict(some_dict):
    count =0
    for key in some_dict.keys():
        count+= len(some_dict[key])
    return count

def is_word_in_dicts_list(word,dicts_list):
    pos = 0
    for dict in dicts_list:
        if word==dict["word"]:
            return True,pos
        pos+=1
    return False,pos

def merge_meta_from_one_to_onother(input_meta,ouput_meta):
    for i in range(len(input_meta["contexts"])):
        ouput_meta["contexts"].append(input_meta["contexts"][i])
    # проверять контексты одного и то го же слова но в разной форме на совпадение?

def singuralize_dict(words_dict):
    dict_keys = words_dict.keys() #parts_of_speech
    base_name = 'NN'
    output={}
    for dict_key in words_dict.keys():
        output.update({dict_key:[]})
    names_for_migrating_from_to_base_name=['NNS', 'NNPS']
    for name in names_for_migrating_from_to_base_name:
        if (name in dict_keys):
            # преобразоавли значения
            values = [one_dict["word"] for one_dict in words_dict[name]]
            transformed_values = [singularize(plural) for plural in values]

            poped_word_dicts_list = words_dict.pop(name, None)

            # миграция из name в base_name с учетом коллизии
            values_in_base_data = [one_dict["word"] for one_dict in words_dict[base_name]]
            for i in range(len(transformed_values)):
                is_in,pos = is_word_in_dicts_list(transformed_values[i],words_dict[base_name])
                if is_in:
                    # если есть коллизия, то объединить только метаданные.без изменения имени
                    merge_meta_from_one_to_onother(poped_word_dicts_list[i]["meta"],words_dict[base_name][pos]["meta"])
                else:
                    #если нету коллизии, то просто перенести метаданные с новым названием слова
                    dict_to_other_cell = poped_word_dicts_list[i]
                    dict_to_other_cell["word"] = transformed_values[i]
                    words_dict[base_name].append(dict_to_other_cell)
    return words_dict

def to_base_verb(words_dict):
    dict_keys= words_dict.keys()  #parts_of_speech
    base_name = 'VB'
    names_for_migrating_from_to_base_name = ['VBD', 'VBG','VBN','VBP','VBZ']

    for name in names_for_migrating_from_to_base_name:
        if (name in dict_keys):
            # преобразоавли значения
            values = [one_dict["word"] for one_dict in words_dict[name]]
            transformed_values = [WordNetLemmatizer().lemmatize(verb_in_other_form, 'v') for verb_in_other_form in values]

            poped_word_dicts_list = words_dict.pop(name, None)

            # миграция из name в base_name с учетом коллизии
            values_in_base_data = [one_dict["word"] for one_dict in words_dict[base_name]]
            for i in range(len(transformed_values)):
                is_in,pos = is_word_in_dicts_list(transformed_values[i],words_dict[base_name])
                if is_in:
                    # если есть коллизия, то объединить только метаданные.без изменения имени
                    merge_meta_from_one_to_onother(poped_word_dicts_list[i]["meta"],words_dict[base_name][pos]["meta"])
                else:
                    #если нету коллизии, то просто перенести метаданные с новым названием слова
                    dict_to_other_cell = poped_word_dicts_list[i]
                    dict_to_other_cell["word"] = transformed_values[i]
                    words_dict[base_name].append(dict_to_other_cell)

    return words_dict

def check_is_it_word_and_delete_if_not(words_dict):
    print("\t\t not word:\n")
    items= words_dict.items()
    i=0
    num_of_items= len(items)
    dict = enchant.Dict("en_US")
    output_dict = {}
    for key,list_of_dicts in items:
        print("\r{}%".format(float(i)/(num_of_items-1)*100.0),end='')
        output_dict.update({key:[]})
        output_dicts = []
        for word_dict in list_of_dicts:
            # проверка является ли слово английским словом
            # if word in words_dataset_from_nltk.words():
            if dict.check(word_dict["word"]):
                output_dicts.append(word_dict)
        output_dict[key] = output_dicts
        i+=1
    print('\n')
    return output_dict

def get_unpopular_words_only(words_dict,MAX_NUM_OF_WORDS_IN_OUTPUT):

    num_of_elements = get_num_of_element_in_dict(words_dict)
    if num_of_elements<=MAX_NUM_OF_WORDS_IN_OUTPUT:
        sections = {"": words_dict}
        return sections
    else:
        print("sorting words by freq")
        # here the problem
        words=[]
        words_meta=[]
        words_types= []
        for key,list_of_dicts in words_dict.items():
            for word_dict in list_of_dicts:
                words.append(copy.deepcopy(word_dict["word"]))
                words_types.append(copy.deepcopy(key))
                words_meta.append(copy.deepcopy((word_dict["meta"])))

        freqs=[]
        for word in words:
            freqs.append(word_frequency(word,'en'))


        argsort = np.argsort(freqs)
        s_words = (np.asarray(words)[argsort]).tolist()
        s_words_types =(np.asarray(words_types)[argsort]).tolist()
        s_words_meta = (np.asarray(words_meta)[argsort]).tolist()

        popular_words_types = s_words_types[MAX_NUM_OF_WORDS_IN_OUTPUT:]
        popular_words = s_words[MAX_NUM_OF_WORDS_IN_OUTPUT:]
        popular_words_meta = s_words_meta[MAX_NUM_OF_WORDS_IN_OUTPUT:]

        unpopular_words_types = s_words_types[:MAX_NUM_OF_WORDS_IN_OUTPUT]
        unpopular_words = s_words[:MAX_NUM_OF_WORDS_IN_OUTPUT]
        unpopular_words_meta = s_words_meta[:MAX_NUM_OF_WORDS_IN_OUTPUT]

        popular_dict ={}
        unpopular_dict={}
        for key, value in words_dict.items():
            popular_dict.update({key:[]})
            unpopular_dict.update({key:[]})
        for i in range(len(popular_words)):
            popular_dict[popular_words_types[i]].append({"word":popular_words[i],"meta":popular_words_meta[i]})
        for i in range(len(unpopular_words)):
            unpopular_dict[unpopular_words_types[i]].append({"word":unpopular_words[i],"meta":unpopular_words_meta[i]})

        popular_dict={k:v for k,v in popular_dict.items() if v}
        unpopular_dict={k:v for k,v in unpopular_dict.items() if v}

    sections = {"ordinary words":popular_dict,"unusual words":unpopular_dict}
    return sections

key_translation_dict =\
    {
        'FW':"foreign word",
        'JJ':"adjective - прилагательное",
        'JJR':"adjective, comparative",
        'JJS':"adjective, superlative",
        'MD':"modal",
        'NN':"singular noun",
        'NNS':"plural noun",
        'PDT':"predeterminer",
        'POS':"possessive ending parent’s",
        'RB':"adverb - наречие",
        'RBR':"adverb, comparative",
        'RBS':"adverb, superlative",
        'RP':"particle",
        'VB':"verb, base form",
        'VBD':"verb, past tense",
        'VBG':"verb, gerund/present participle",
        'VBN':"verb, past participle",
        'VBP':"verb, sing. present, non-3d",
        'VBZ':"verb, 3rd person sing. present"
    }

def remane_keys(some_dict):
    output_dict ={}
    for key, value in some_dict.items():
        output_dict.update({key_translation_dict[key]:value})
    return output_dict



def get_words_contexts(pdf_path,words_dict):
    print("\tget words contexts")
    output_dict = {}
    for key in words_dict.keys():
        output_dict.update({key:[]})
    with fitz.open(pdf_path) as pdf:
        num_of_pages = len(pdf)
        for i, page in enumerate(pdf):
            print("\r{}%".format(float(i) / (num_of_pages - 1) * 100.0), end='')
            blocks_in_page = page.get_text("blocks")
            for block in blocks_in_page:
                text_of_block = block[4]
                tokenized_block_into_sents = sent_tokenize(text_of_block)
                for sentence in tokenized_block_into_sents:
                    # считаем sentence контекстом слова, если это слово есть в sentence
                    for key, value in words_dict.items():
                        for word in value:
                            if word in sentence:
                                # сразу уникализуем вхождения
                                is_in, pos = is_word_in_dicts_list(word,output_dict[key])
                                if is_in:
                                    output_dict[key][pos]['meta']['contexts'].append(sentence)
                                else:
                                    output_dict[key].append({'word':word,'meta':{'contexts':[sentence]}})
    return output_dict

def glue_line_breaks(tokens):
    output=[]
    on_line_brake = False
    for i in range(len(tokens)-1):
        if on_line_brake:
            on_line_brake=False
            continue
        if len(tokens[i])>1 and len(tokens[i+1])>1:
            if tokens[i][-1] == '-' and tokens[i+1][0].islower():
                output.append(tokens[i][:-1]+tokens[i+1])
                on_line_brake=True
                continue
        output.append(tokens[i])
    if not on_line_brake:
        output.append(tokens[-1])
    return output

def run_backend_get_all_unique_words(source_pdf_path,
                                     MAX_NUM_OF_OUTPUT=300):
    print("getting unique words")
    start_time = time.time()
    pdf_path= source_pdf_path
    table_for_replacing_abbreviations = \
        {
            "fig": "figure"
        }
    english_alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                        "T",
                        "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                        "n",
                        "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


    clear_dir("./pixmaps")

    # get all unique words

    all_words_dict = {}
    # get all row "english words"
    stop_words = set(stopwords.words('english'))

    # эти части речи попадут не будут отфильтрованы
    allowed_parts_of_speech = ['FW','JJ','JJR','JJS','MD','NN','NNS','PDT','POS','RB','RBR','RBS','RP','VB','VBD','VBG','VBN','VBP','VBZ']
    for part_of_speech in allowed_parts_of_speech:
        all_words_dict.update({part_of_speech:[]})

    # make filters of "words" also
    disallowed_characters = "!@#$%^&*()_+1234567890-–=<>?,./`~”“№;:[]{}•’"
    replacing_keys_ = table_for_replacing_abbreviations.keys()

    print("\treading row text")
    with fitz.open(pdf_path) as pdf:
        num_of_pages = len(pdf)
        for i, page in enumerate(pdf):
            print("\r{}%".format(float(i)/(num_of_pages-1)*100.0),end='')
            blocks_in_page = page.get_text("blocks")
            for block in blocks_in_page:
                text_of_block = block[4]
                tokenized_block_into_sents = sent_tokenize(text_of_block)
                for sentence in tokenized_block_into_sents:

                    # здесь должен быть фильтр уравнений или математических выражений

                    # получаем последовательнсот слов из предложения
                    word_list = nltk.word_tokenize(sentence,"english",True)
                    # обрабатываем перенос строки с помощью "-"
                    word_list = glue_line_breaks(word_list)
                    # убираем спец слова
                    word_list = [w for w in word_list if w not in stop_words]
                    # убираем не нужные символы
                    word_list = filter_word_list(word_list,filter_of_trash_characters,
                    disallowed_characters=disallowed_characters,
                    english_alphabet=english_alphabet)

                    # помечаем частями речи
                    tagged_words = nltk.pos_tag(word_list)
                    # убираем не нужные части речи
                    tagged_words = filter_tagged_words(tagged_words,filter_for_tagged_words,
                                                       parts_of_speech=allowed_parts_of_speech)
                    # пишем полученную последовательность в выходной поток
                    for tagged_word in tagged_words:
                        all_words_dict[tagged_word[1]].append(tagged_word[0].lower())
        print("\n")

    print("\trow text processing")
    #уникализуем слова в словаре
    all_words_dict = refresh_all_word_dict(all_words_dict)

    # возьмем контекст слов
    all_words_dict = get_words_contexts(pdf_path,all_words_dict)


    # переведем существительные в единственное число и отбросим повторы
    all_words_dict = singuralize_dict(all_words_dict)
    # переведем глаголы в базовую форму и отбросим повторы
    all_words_dict = to_base_verb(all_words_dict)
    all_words_dict = check_is_it_word_and_delete_if_not(all_words_dict)#
    all_words_dict = remane_keys(all_words_dict)

    # далее не переписано

    # из полученных слов сделаем два словаря с популярными и непопулярными словами
    sections = get_unpopular_words_only(all_words_dict,MAX_NUM_OF_WORDS_IN_OUTPUT=MAX_NUM_OF_OUTPUT)

    stop_time= time.time()
    print("\ttime of processing {} s".format(stop_time-start_time))
    return sections

def run_backend_get_tranlate_of_unique_words(words_dict):
    traslate_output = {}
    driver = webdriver.Chrome()
    driver.get('https://translate.yandex.ru/')

    input = driver.find_element(By.ID, "fakeArea")
    output = driver.find_element(By.ID, "translation")
    i=0
    for key,words_dicts_list in words_dict.items():
        traslate_output.update({key:[]})
        for word_dict in words_dicts_list:
            input.clear()
            input.send_keys(word_dict["word"])
            input.send_keys(Keys.RETURN)
            time.sleep(2)
            word_translation = output.get_attribute("value")
            example_group = driver.find_elements(By.CLASS_NAME, "example_group")
            other =""
            for group in example_group:
                other+=group.text
            # translate some of source_context:
            source_contexts_translation = []
            max_num_of_source_contexts_tranlation = 10
            if max_num_of_source_contexts_tranlation > len(word_dict["meta"]["contexts"]):
                max_num_of_source_contexts_tranlation = len(word_dict["meta"]["contexts"])
            for j in range(max_num_of_source_contexts_tranlation):
                # берем первые попавшиеся. без случайности
                input.clear()
                input.send_keys(word_dict["meta"]["contexts"][j])
                input.send_keys(Keys.RETURN)
                time.sleep(3)
                text_translation = output.get_attribute("value")
                source_contexts_translation.append({"en": word_dict["meta"]["contexts"][j], "ru":text_translation})

            # get ya_context for [{"en":,"ru":}]
            ya_contexts = []
            max_num_of_ya_contexts_tranlation = 10
            if len(other)!=0:
                splitted = other.split('\n')
                #if len(splitted)%2 != 0: # do nothing, but it can be problem
                if len(splitted)<max_num_of_ya_contexts_tranlation:
                    max_num_of_ya_contexts_tranlation=len(splitted)
                for index_of_sent in range(0,max_num_of_ya_contexts_tranlation,2):
                    ya_contexts.append({"en":splitted[index_of_sent],"ru":splitted[index_of_sent+1]})

            traslate_output[key].append({"word":{"en":word_dict["word"],"ru":word_translation}, "ya_context":ya_contexts,"source_contexts":source_contexts_translation})
            print("{}/{} word [{}] translate [{}]".format(i,300,word_dict["word"],word_translation))
            i+=1
    driver.close()
    return traslate_output


def transpose_list(x):
    output=[]
    for i in range(len(x)):
        output.append([x[i]])
    return output

def make_h1(str,tabulation):
    output_str=''
    output_str+='<font size="1" face="Times New Roman" >\n'
    output_str+=tabulation+'<h1 align="center">'+'<span style="font-weight: normal;">'+str +'</span>'+'</h1>\n'
    output_str+=tabulation+'</font>\n'
    return output_str

def make_h2(str,tabulation):
    return tabulation + '<h2 align="center">' + '<span style="font-weight: normal;">'+str +'</span>'+ '</h2>\n'


def make_table(rows,header,tabulation):
    table_str = '<font size="2" face="Times New Roman" >\n'
    # table_str +=tabulation+'<table class="center">\n'
    table_str +=tabulation+'<table class="center" width="50%">\n'
    # make headers of columns
    # table_str+=tabulation+"\t<tr>\n"
    # for header_of_column in headers:
    table_str+=tabulation+make_h2(header,'\t\t\t')
    # table_str+=tabulation+"\t</tr>\n"

    for row in rows:
        if len(row)==1:
            table_str += tabulation + '\t<tr>\n'
            table_str += tabulation + '\t\t<td colspan="2">\n'
            table_str += tabulation + '\t\t\t' + row[0] + '\n'
            table_str += tabulation + '\t\t</td>\n'
            table_str += tabulation + '\t</tr>\n'
        else:
            table_str += tabulation + '\t<tr>\n'
            for value in row:
                table_str += tabulation + '\t\t<td>\n'
                table_str += tabulation+'\t\t\t'+'<p><b>'+value+'</b></p>\n'
                table_str += tabulation + '\t\t</td>\n'
            table_str += tabulation + '\t</tr>\n'


    table_str+=tabulation+'</table>\n'
    table_str += '</font>\n'
    return table_str





def run_fronted_add_table_with_unique_words_to_html(translate,output_path):
    html_code = ''
    html_code+=\
"""
<!DOCTYPE html>
<html>

    <head>
        <meta charset="utf-8">
        <style>
        .center {
            margin-left: auto;
            margin-right: auto;
        }

        table, th, td {
        border: 1px solid black;
        }
        </style>
    </head>
    
    <body>
"""
    html_code += make_h1('overview of words', '\t\t')
    # traslate_output[key].append({"word": {"en": word_dict["word"], "ru": word_translation}, "ya_context": ya_contexts,
    #                              "source_contexts": source_contexts_translation})

    for words_type, words_list in translate.items():
        rows = []
        for word_dict in words_list:
            eng_field= word_dict["word"]["en"]
            ru_field = word_dict["word"]["ru"]
            if len(ru_field)>0:
                rows.append([eng_field,ru_field])
            else:
                rows.append([eng_field])
            source_contexs = word_dict["source_contexts"]
            for source_context in source_contexs:
                rows.append([source_context["en"]])
                if len(source_context["ru"])>0:
                    rows.append([source_context["ru"]])

            ya_contexts = word_dict["ya_context"]
            for ya_context in ya_contexts:
                if len(ya_context["en"])>0:
                    rows.append([ya_context["en"]])
                if len(ya_context["ru"])>0:
                    rows.append([ya_context["ru"]])
        html_code+=make_table(rows,words_type,"\t\t\t")


    html_code+=\
"""
    </body>
    
</html>
"""
    # with open(output_path,'wb') as out:
    #     out.write(html_code.encode('utf-8'))
    with codecs.open(output_path, 'wb', encoding='utf-8') as out:
        out.write(html_code)

#
#
# raise SystemExit
#
# # get all unique english words
#
# unique_row_strings = list(set(all_row_strings))
# enc_dict = enchant.Dict("en_US")
# disallowed_characters = "!@#$%^&*()_+1234567890-=<>?,./`~"
# replacing_keys_ = table_for_replacing_abbreviations.keys()
#
#
# engish_words = []
# for row_string in unique_row_strings:
#     copy_of_row_string = copy(row_string)
#     pure_row = copy_of_row_string.translate({ord(c): None for c in disallowed_characters})
#     if (pure_row not in english_alphabet):
#         if (pure_row != ""):
#             # добавить проверку на вхождения "ii","vii" и тому подобные - приходит из математических выражений
#             if (enc_dict.check(pure_row)):
#                 engish_words.append(replace_if_in_replacement_table(pure_row.lower(),replacing_keys_,table_for_replacing_abbreviations))
#
#
#
# # pluralize all unique english words
# plural_maker = Pluralizer()
# plural_unique_words = list(set([plural_maker.singular(plural) for plural in engish_words]))
#
# #reduce all personal pronoun, coordinating conjunction,
#
#
# print("number of unique words {} in pdf {}".format(len(plural_unique_words), pdf_path))
# pprint.pprint(plural_unique_words)
#
# #  my pages info container
# pages_contents = {}

# table_htmp_code = tabulate(transpose_list(words_list),headers=[words_type],tablefmt="html")
# print(table_htmp_code)
# with open(output_path, 'w') as text_file:
    # text_file.write(table_htmp_code)