# 元组序列为<动作，介词，动作对象，动作对象的属性，动作输入>
import time

import gensim
import spacy
import stanza
from spacy.lang.char_classes import LIST_ELLIPSES, LIST_ICONS, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, ALPHA
from spacy.symbols import ORTH, LEMMA, POS, TAG, NORM
from gensim.models import word2vec
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import bs4
import lxml
import re
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import compile_infix_regex
from stanza.models.common import doc
from stanza.pipeline._constants import TOKENIZE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

# 自动化每一步等待时间
sleep_time = 10
# 开始处理号和结束号
bug_begin = 1
bug_end = 2
# 自动化开关
open_driver = False
# 加载word2vec 模型
model = gensim.models.Word2Vec.load('./model.model')


def my_en_tokenizer_url(nlp):  # 自然语言处理（网址分词）
    prefix_re = re.compile(r'''^[\[\({}"']''')
    suffix_re = re.compile(r'''[\]\){}"'.,]$''')
    infix_re = re.compile(r'''[~]''')
    pattern_re = re.compile(r'^https?://|^`|^"|^\'')

    return spacy.tokenizer.Tokenizer(nlp.vocab,
                                     English.Defaults.tokenizer_exceptions,
                                     prefix_re.search,
                                     suffix_re.search,
                                     infix_re.finditer,
                                     token_match=pattern_re.match)


@register_processor_variant(TOKENIZE, 'spacy_own')
class SpacyTokenizer(ProcessorVariant):
    def __init__(self, config):
        """ Construct a spaCy-based tokenizer by loading the spaCy pipeline.
        """
        if config['lang'] != 'en':
            raise Exception("spaCy tokenizer is currently only allowed in English pipeline.")

        try:
            import spacy
            from spacy.lang.en import English
        except ImportError:
            raise ImportError(
                "spaCy 2.0+ is used but not installed on your machine. Go to https://spacy.io/usage for installation instructions."
            )

        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        # self.nlp = spacy.blank("en")
        self.nlp = English()
        # by default spacy uses dependency parser to do ssplit
        # we need to add a sentencizer for fast rule-based ssplit
        if spacy.__version__.startswith("2."):
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
        else:
            self.nlp.add_pipe("sentencizer")

        # self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)
        self.nlp.tokenizer = my_en_tokenizer_url(self.nlp)
        # 特殊分词
        self.nlp.tokenizer.add_special_case("that's", [{"ORTH": "that's"}])
        self.no_ssplit = config.get('no_ssplit', False)

    def process(self, document):
        """ Tokenize a document with the spaCy tokenizer and wrap the results into a Doc object.
        """
        if isinstance(document, doc.Document):
            text = document.text
        else:
            text = document
        if not isinstance(text, str):
            raise Exception("Must supply a string or Stanza Document object to the spaCy tokenizer.")
        spacy_doc = self.nlp(text)

        # 合并特殊情况（代码情况）
        # "情况
        double_quotes_search = re.compile('''"[^"|^ ]+[^"]*[^"|^ ]+"''')
        # `情况
        inclined_quote_search = re.compile('''`[^`|^ ]+[^`]*[^`|^ ]+`''')
        # '情况
        one_quote_search = re.compile("'[^'|^ ]+[^']*[^'|^ ]+'")
        # ()情况
        small_bracket_search = re.compile('''\([^(|^)]+\)''')
        # {}情况
        large_bracket_search = re.compile('''{[^(|^)]+}''')
        # url:情况
        url_maohao_search = re.compile('''(?<=url: )[^ ]+''')
        # payload:情况
        payload_maohao_search_1 = re.compile('''(?<=payload: ).+''')
        payload_maohao_search_2 = re.compile('''(?<=payload : ).+''')
        # 找寻匹配项
        search_sentences = double_quotes_search.findall(text) + inclined_quote_search.findall(
            text) + small_bracket_search.findall(text) + large_bracket_search.findall(text) + \
                           one_quote_search.findall(text) + url_maohao_search.findall(text) + \
                           payload_maohao_search_1.findall(text) + payload_maohao_search_2.findall(text)
        for sentence in search_sentences:
            special_text = self.nlp(sentence)
            special_text_real = [word.text for word in special_text]
            max_index = -1
            min_index = 9999
            now_index = 0
            for i, word in enumerate(spacy_doc):
                if word.text == special_text_real[now_index]:
                    if now_index == 0:
                        min_index = i
                    max_index = i
                    now_index += 1
                else:
                    now_index = 0
                if now_index >= len(special_text_real):
                    break
            # 没找到或者本身就是一个的话不做处理
            if max_index != -1 and max_index != min_index:
                with spacy_doc.retokenize() as retokenizer:
                    retokenizer.merge(spacy_doc[min_index:max_index + 1])  # 合并初步处理过的数据，把多个单词按要求合并为一个

        sentences = []
        for sent in spacy_doc.sents:
            tokens = []
            for tok in sent:
                token_entry = {
                    doc.TEXT: tok.text,
                    doc.MISC: f"{doc.START_CHAR}={tok.idx}|{doc.END_CHAR}={tok.idx + len(tok.text)}"
                }
                tokens.append(token_entry)
            sentences.append(tokens)

        # if no_ssplit is set, flatten all the sentences into one sentence
        if self.no_ssplit:
            sentences = [[t for s in sentences for t in s]]

        return doc.Document(sentences, text)


nlp = stanza.Pipeline('en', r"C:\Users\Administrator\stanza_resources",
                      processors={'tokenize': 'spacy_own'})  # 指定自定义的分词器（分词管道）

tag_of_obj_1 = ['NN', 'NNP', 'NNS']
tag_of_obj_2 = ['.', 'FW', 'NNP']
head_of_url = ['http', 'www.']
tag_of_prep = ['TO', 'IN']
dep_of_prep = ['prep', 'prt', 'aux', 'case']
dep_of_obj = ['dobj']
dep_of_mod_1 = ['nmod', 'amod', 'compound']
dep_of_mod_2 = ['relcl']
dep_conj = ['conj', 'parataxis']
dep_compound = ['compound']

http_login_verb = ['go', 'login', 'navigate', 'navigating', 'visit', 'triggered', 'using']  # 页面跳转
http_login_dep = ['obj', 'parataxis', 'obl']

login_verb = ['go', 'login', 'logged', 'using', 'navigate', 'log']  # 前往
login_xpos = ['NN', 'NNS', 'NNP']
login_dep_dobj = ['obj', 'obl']
login_dep_mod = ['nmod', 'amod']

update_verb = ['update', 'upload', 'save']  # 更新，上传
update_dep_dobj = ['obj']

input_verb = ['enter', 'type', 'add', 'write', 'paste', 'put', 'insert', 'fill', 'adding']  # 输入
input_dep_dobj = ['dobj', 'obj']
input_dep_pobj = ['pobj', 'obl']
input_dep_mod_1 = ['amod', 'compound', 'appos']
input_dep_mod_2 = ['amod', 'compound', 'appos']

use_verb = ['use']  # 使用
use_dep_dobj = ['dobj', 'obj']
use_dep_pobj = ['pobj', 'obl']
use_dep_mod_1 = ['amod', 'compound']
use_dep_mod_2 = ['compound', 'appos']
use_dep_mod_3 = ['advmod']

click_verb = ['click']  # 点击
click_dep_pobj = ['pobj', 'obl', 'dep', 'advmod']
click_dep_mod = ['nmod', 'amod', 'compound']

change_verb = ['change']  # 修改
change_dep_dobj = ['obj']
change_dep_pobj = ['obl']
change_dep_mod = ['nmod', 'amod']

select_verb = ['select']  # 选择
select_dep_dobj = ['obj']
select_dep_pobj = ['pobj', 'advcl']
select_dep_mod = ['nmod', 'amod']

capture_verb = ['capture']  # 粘贴
capture_dep_dobj = ['obj']
capture_dep_mod = ['nmod', 'amod']

role_of_prp = ['you']  # 断言名称

verb_type = {  # 动词类型
    'skip_url': 1,  # url跳转
    'input_verb': 2,  # 带输入型
    'no_input_verb': 3,  # 不带输入型
}

payload_find_1 = re.compile("<[^>|^ ]+[^>]*[^>|^ ]+>")
payload_find_2 = re.compile("<script>[^ ]+[^>]*[^ ]+<\/script>")
payload_find_3 = re.compile("(?<=Payload: ).+and")


# 加载模型
# model = word2vec.Word2Vec.load('C:\Users\Administrator\Desktop\bug\model.model')


def print_relation_of_step(doc):  # 输出关系图
    print("{:<15} | {:<10} | {:<10} | {:<15} ".format('Token', 'POS', 'Relation', 'Head'))
    print("-" * 0)

    # 将句子对象转换为字典
    sent_dict = doc.sentences[0].to_dict()
    for word in sent_dict:
        print("{:<15} | {:<10} |{:<10} | {:<15} "
              .format(str(word['text']), str(word['xpos']), str(word['deprel']),
                      str(sent_dict[word['head'] - 1]['text'] if word['head'] > 0 else 'ROOT')))
    # for token in doc:
    #     print('{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.xpos, token.deprel, doc[token.head-1].text
    #                                                 , doc[token.head-1].xpos))


def catch_root(doc):
    for token in doc:
        if token.deprel == 'ROOT':
            return token.text


def count_verb(doc, verb_count, verb_list):  # 计算动作数
    # for token in doc:
    #     if token.deprel == 'ROOT':
    #         root_verb = token.text
    for token in doc:
        if 'VB' in token.xpos and (
                'root' == token.deprel or (token.deprel in dep_conj and doc[token.head - 1].text.lower() in verb_list)):
            verb_list.append(token.text.lower())
            verb_count += 1
        if token.xpos == 'VBG' and doc[
            token.head - 1].text.lower() in verb_list and token.deprel in 'advcl':  # 处理by + verbing
            for index, verb in enumerate(verb_list):
                if verb == doc[token.head - 1].text.lower():
                    verb_list[index] = token.text.lower()
    return verb_count


def has_assert(doc, var_lists):  # 处理断言
    for token in doc:
        if token.text in role_of_prp:  # 如果包含主语
            var_lists[0] = 1
            break


# def catch_verb(doc, now_of_count, var_list, verb_list):  # 获得动词，五元组第一
#     for token in doc:
#         if 'VB' in token.tag_ and doc[token.head-1].text in verb_list:
#             if now_of_count == 0:
#                 var_list[0] = token.text
#                 break
#             now_of_count -= 1
#             continue


def catch_prep(doc, now_of_count, var_list, verb_list):  # 找到介词，五元组第二
    now_is_which = -1  # 现在是第几个
    for token in doc:
        if token.text.lower() in verb_list:
            now_is_which += 1
        if now_is_which == now_of_count and token.deprel in dep_of_prep:
            var_list[1] = token.text
            break


def catch_obj(doc, now_of_count, var_list, verb_list, have_http):  # 找到宾语，五元组第三
    third_list = []
    now_is_which = -1  # 现在是第几个
    for token in doc:
        if token.text.lower() in verb_list:
            now_is_which += 1
        if now_is_which == now_of_count:
            if verb_list[now_is_which] in http_login_verb and token.deprel in http_login_dep and token.text[0:4] \
                    in head_of_url:  # go url
                if len(third_list) >= 1:
                    third_list[0] = token.text
                    break
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in login_verb and token.deprel in login_dep_dobj and token.xpos in login_xpos and \
                    token.text[0:4] not in head_of_url:  # go but not go url
                third_list.append(token.text)
                if not have_http:  # 如果有网址这里先不退出，因为网址在这后面的话会先判断这个导致网址没获取到，但是网址在这前面的话就不会先到这里了
                    break
            if verb_list[now_is_which] in update_verb and token.deprel in update_dep_dobj and token.text[0:4] \
                    not in head_of_url:  # update
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in input_verb and token.deprel in input_dep_pobj and doc[
                token.head - 1].text.lower() == \
                    var_list[0]:  # input
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in use_verb and token.deprel in use_dep_pobj:  # use
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in click_verb and token.deprel in click_dep_pobj \
                    and doc[token.head - 1].text.lower() == var_list[0]:  # click
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in change_verb and token.deprel in change_dep_dobj and doc[
                token.head - 1].text.lower() == \
                    var_list[0]:  # change
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in select_verb and token.deprel in select_dep_dobj and doc[
                token.head - 1].text.lower() == \
                    var_list[0]:  # select
                third_list.append(token.text)
                break
            if verb_list[now_is_which] in capture_verb and token.deprel in capture_dep_dobj and doc[
                token.head - 1].text.lower() == \
                    var_list[0]:  # capture
                third_list.append(token.text)
                break
            # if token.tag_ in tag_of_obj_1 and doc[token.head-1].text == var_list[1]:  # not url
            #     var_list[2] = token.text
            # if token.tag_ in tag_of_obj_2 and doc[token.head-1].text == var_list[0]:  # url
            #     var_list[2] = token.text
            #     break
    for token in doc:
        if token.deprel in dep_conj and doc[token.head - 1].text in third_list:
            third_list.append(token.text)
    var_list[2] = third_list


def catch_putin(doc, now_of_count, var_list, verb_list):  # 找输入，五元组第五
    fifth_list = []
    now_is_which = -1  # 现在是第几个
    if verb_list[now_is_which] not in http_login_verb:  # not url
        for token in doc:
            if token.text.lower() in verb_list:
                now_is_which += 1
            if now_is_which == now_of_count:
                if verb_list[now_is_which] in input_verb and token.deprel in input_dep_dobj and doc[
                    token.head - 1].text.lower() == \
                        var_list[0]:  # input
                    fifth_list.append(token.text)
                    break
                if verb_list[now_is_which] in use_verb and token.deprel in use_dep_dobj and doc[
                    token.head - 1].text.lower() == \
                        var_list[0]:  # use
                    fifth_list.append(token.text)
                    break
                if verb_list[now_is_which] in change_verb and token.deprel in change_dep_pobj and doc[
                    token.head - 1].text.lower() == \
                        var_list[0]:  # change
                    fifth_list.append(token.text)
                    break
        for token in doc:
            if token.deprel in dep_conj and doc[token.head - 1].text in fifth_list:
                fifth_list.append(token.text)
        var_list[4] = fifth_list


def catch_adj_of_obl(doc, now_of_count, var_list, verb_list):  # 找动词对象的形容词，五元组第四
    now_is_which = -1  # 现在是第几个
    forth_list = []
    # if verb_list[now_of_count] not in http_login_verb:  # not url
    for token in doc:
        if token.text.lower() in verb_list:
            now_is_which += 1
        if now_of_count == now_is_which:
            if verb_list[now_is_which] in input_verb and token.deprel in input_dep_mod_1 and doc[token.head - 1].text in \
                    var_list[2]:  # input
                forth_list.append(token.text)
            if verb_list[now_is_which] in use_verb and token.deprel in use_dep_mod_1 and doc[token.head - 1].text in \
                    var_list[2]:  # use
                forth_list.append(token.text)
            if verb_list[now_is_which] in click_verb and token.deprel in click_dep_mod and doc[token.head - 1].text in \
                    var_list[2]:  # click
                forth_list.append(token.text)
            if verb_list[now_is_which] in change_verb and token.deprel in change_dep_mod and doc[token.head - 1].text in \
                    var_list[2]:  # change
                forth_list.append(token.text)
            if verb_list[now_is_which] in login_verb and token.deprel in login_dep_mod and doc[token.head - 1].text in \
                    var_list[2]:  # go but not url
                forth_list.append(token.text)
            if verb_list[now_is_which] in capture_verb and token.deprel in capture_dep_mod and doc[
                token.head - 1].text in \
                    var_list[2]:  # capture
                forth_list.append(token.text)
            # if (token.dep_ in dep_of_mod_1 and doc[token.head-1].text == var_list[2]) or (
            #         token.dep_ in dep_of_mod_2 and doc[token.head-1].text == var_list[4]):
            #     var_list[3] = token.text
            #     break
    for token in doc:
        if token.deprel in dep_conj and doc[token.head - 1].text in forth_list:
            forth_list.append(token.text)
    var_list[3] = forth_list


def catch_adj_of_obj(doc, now_of_count, var_list, verb_list, text):  # 找介词对象的形容词，五元组第六
    now_is_which = -1  # 现在是第几个
    sixth_list = []
    for token in doc:
        if token.text.lower() in verb_list:
            now_is_which += 1
        if now_of_count == now_is_which:
            if verb_list[now_is_which] in input_verb and (
                    (token.deprel in input_dep_mod_2 and doc[token.head - 1].text in var_list[4]) or (
                    token.deprel in input_dep_pobj and doc[token.head - 1].text.lower() in var_list[
                0] and '''"''' in token.text)):  # input
                sixth_list.append(token.text)
            if verb_list[now_is_which] in use_verb and ((token.deprel in use_dep_mod_2 and doc[token.head - 1].text in \
                                                         var_list[4]) or (token.deprel in use_dep_mod_3 and doc[
                token.head - 1].text.lower() in var_list[0])):  # use
                sixth_list.append(token.text)
            if verb_list[now_is_which] in click_verb and token.deprel in click_dep_mod and doc[token.head - 1].text in \
                    var_list[4]:  # click
                sixth_list.append(token.text)
            if verb_list[now_is_which] in change_verb and token.deprel in change_dep_mod and doc[token.head - 1].text in \
                    var_list[4]:  # change
                sixth_list.append(token.text)
            if verb_list[now_is_which] in login_verb and token.deprel in login_dep_mod and doc[token.head - 1].text in \
                    var_list[4]:  # go but not url
                sixth_list.append(token.text)
            if verb_list[now_is_which] in capture_verb and token.deprel in capture_dep_mod and doc[
                token.head - 1].text in \
                    var_list[4]:  # capture
                sixth_list.append(token.text)
    for token in doc:
        if token.deprel in dep_conj and doc[token.head - 1].text in sixth_list:
            sixth_list.append(token.text)
    if ('payload' in var_list[4] or 'Payload' in var_list[4]) and len(sixth_list) == 0:
        for index, sentence in enumerate(text):
            if 'payload' in sentence.lower() and 'step' not in sentence.lower():
                payload_sentence = text[index] + text[index + 1]
                payload_findall_1 = payload_find_1.findall(payload_sentence)
                payload_findall_2 = payload_find_2.findall(payload_sentence)
                payload_findall_3 = payload_find_3.findall(payload_sentence)
                if len(payload_findall_3) > 0:
                    for find_index, find_sentence in enumerate(payload_findall_3):
                        payload_findall_3[find_index] = payload_findall_3[find_index][
                                                        0:len(payload_findall_3[find_index]) - 4]
                    sixth_list = payload_findall_3
                if len(payload_findall_2) > 0:
                    sixth_list = payload_findall_2
                else:
                    if len(payload_findall_1) > 0:
                        sixth_list = payload_findall_1
                print('1111111')
                print(sixth_list)
    var_list[5] = sixth_list


def has_nothing(var_list):
    if var_list is None or len(var_list) == 0:
        return True
    return False


# 词语还原
def re_words(word):
    if word[len(word) - 1] == '''"''' or word[0] == '''"''':
        return word.strip('''"''')
    if word[len(word) - 1] == "'" or word[0] == "'":
        return word.strip("'")
    if word[len(word) - 1] == "=" or word[0] == "=":
        return word.strip("=")
    if word[len(word) - 1] == ")" and word[0] == "(":
        return word.strip("()")
    if word[len(word) - 1] == "}" and word[0] == "{":
        return word.strip("{}")
    return word


# 元组序列为<动作，介词，动作对象，动作对象的类性，动作输入，输入的形容词>
def auto_step(var_list, driver):  # 自动化脚本 每个六元组调一次
    # 2345不空
    if has_nothing(var_list[2]) and has_nothing(var_list[3]) and has_nothing(var_list[4]) and has_nothing(var_list[5]):
        return

    if var_list[0] in http_login_verb:  # url输入
        for words in var_list[2] + var_list[3]:
            word = words.strip('"')
            if word[0:4] in head_of_url:
                driver.get(word)
        time.sleep(10)
        return

    url = driver.current_url
    resp = requests.get(url)  # 爬虫网页
    # print(resp)  # 打印请求结果的状态码
    # print(resp.content)  # 打印请求到的网页源码
    # https://blog.csdn.net/shenyuan12/article/details/108038526
    bs_obj = bs4.BeautifulSoup(resp.content, 'lxml')  # 将网页源码构造成BeautifulSoup对象

    if var_list[0] in input_verb + use_verb:  # input
        input_list = bs_obj.find_all('input')  # 获得所有input类型
        input_id = None
        input_class = None
        input_name = None
        for a in input_list:
            if var_list[3] is not None and len(var_list[3]) > 0:
                for obj_index in var_list[3]:
                    obj_index_without = re_words(obj_index)
                    # for words_of_a in [a.get('name'), a.get('id')] + a.get('class'):
                    #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                    #         input_id = a.get('id')
                    #         input_class = a.get('class')
                    for x in [a.get('name'), a.get('id'), a.get('type')]:
                        if x is not None and obj_index_without in x:
                            input_id = a.get('id')
                            input_class = a.get('class')
                            input_name = a.get('name')
                    if a.get('class') is not None:
                        for x in a.get('class'):
                            if x is not None and obj_index_without in x:
                                input_id = a.get('id')
                                input_class = a.get('class')
                                input_name = a.get('name')

                    if input_id is None and input_name is None and input_class is None:
                        continue
                    if input_id is not None:
                        search_input = driver.find_element(By.ID, input_id)
                    else:
                        if input_name is not None:
                            search_input = driver.find_element(By.NAME, input_name)
                        else:
                            if input_class is not None:
                                search_input = driver.find_element(By.CLASS_NAME, input_class[0])
                    if var_list[5] is not None and len(var_list[5]) > 0:
                        for words_index in var_list[5]:
                            search_input.send_keys(re_words(words_index))
                    else:
                        if var_list[4] is not None and len(var_list[4]) > 0:
                            for words_index in var_list[4]:
                                search_input.send_keys(re_words(words_index))
                    time.sleep(sleep_time)
            else:
                if var_list[2] is not None and len(var_list[2]) > 0:
                    for obj_index in var_list[2]:
                        obj_index_without = re_words(obj_index)
                        # for words_of_a in [a.get('name'), a.get('id')] + a.get('class'):
                        #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                        #         input_id = a.get('id')
                        #         input_class = a.get('class')
                        for x in [a.get('name'), a.get('id'), a.get('type')]:
                            if x is not None and obj_index_without in x:
                                input_id = a.get('id')
                                input_class = a.get('class')
                                input_name = a.get('name')
                        if a.get('class') is not None:
                            for x in a.get('class'):
                                if x is not None and obj_index_without in x:
                                    input_id = a.get('id')
                                    input_class = a.get('class')
                                    input_name = a.get('name')

                        if input_id is None and input_name is None and input_class is None:
                            continue
                        if input_id is not None:
                            search_input = driver.find_element(By.ID, input_id)
                        else:
                            if input_name is not None:
                                search_input = driver.find_element(By.NAME, input_name)
                            else:
                                if input_class is not None:
                                    search_input = driver.find_element(By.CLASS_NAME, input_class[0])
                        if var_list[5] is not None and len(var_list[5]) > 0:
                            for words_index in var_list[5]:
                                search_input.send_keys(re_words(words_index))
                        else:
                            if var_list[4] is not None and len(var_list[4]) > 0:
                                for words_index in var_list[4]:
                                    search_input.send_keys(re_words(words_index))
                        time.sleep(sleep_time)
        return

    if var_list[0] in click_verb + select_verb:  # click
        button_list = bs_obj.find_all('button') + bs_obj.find_all('a')  # 获得所有button类型和a类型
        no_input_id = None
        no_input_class = None
        no_input_name = None
        for a in button_list:
            if var_list[3] is not None and len(var_list[3]) > 0:
                for obj_index in var_list[3]:
                    obj_index_without = re_words(obj_index)
                    # for words_of_a in [a.get('name'), a.get('id'), a.contents[0]] + a.get('class'):
                    #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                    #         no_input_id = a.get('id')
                    #         no_input_class = a.get('class')
                    print('111111111', obj_index_without)
                    print('3333333333 ', [a.get('name'), a.get('id')] + a.contents)
                    for x in [a.get('name'), a.get('id')] + a.contents:
                        if x is not None and obj_index_without in x:
                            no_input_id = a.get('id')
                            no_input_class = a.get('class')
                            no_input_name = a.get('name')
                    if a.get('class') is not None:
                        for x in a.get('class'):
                            if x is not None and obj_index_without in x:
                                no_input_id = a.get('id')
                                no_input_class = a.get('class')
                                no_input_name = a.get('name')

                    if no_input_id is None and no_input_name is None and no_input_class is None:
                        continue
                    if no_input_id is not None:
                        search_button = driver.find_element(By.ID, no_input_id)
                    else:
                        if no_input_name is not None:
                            search_button = driver.find_element(By.NAME, no_input_name)
                        else:
                            if no_input_class is not None:
                                search_button = driver.find_element(By.CLASS_NAME, no_input_class[0])
                    search_button.click()
                    time.sleep(sleep_time)
            else:
                if var_list[2] is not None and len(var_list[2]) > 0:
                    for obj_index in var_list[2]:
                        obj_index_without = re_words(obj_index)
                        # for words_of_a in [a.get('name'), a.get('id'), a.contents[0]] + a.get('class'):
                        #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                        #         no_input_id = a.get('id')
                        #         no_input_class = a.get('class')
                        print('111111111', obj_index_without)
                        print('3333333333 ', [a.get('name'), a.get('id')] + a.contents)
                        for x in [a.get('name'), a.get('id')] + a.contents:
                            if x is not None and obj_index_without in x:
                                no_input_id = a.get('id')
                                no_input_class = a.get('class')
                                no_input_name = a.get('name')
                        if a.get('class') is not None:
                            for x in a.get('class'):
                                if x is not None and obj_index_without in x:
                                    no_input_id = a.get('id')
                                    no_input_class = a.get('class')
                                    no_input_name = a.get('name')

                        if no_input_id is None and no_input_name is None and no_input_class is None:
                            continue
                        if no_input_id is not None:
                            search_button = driver.find_element(By.ID, no_input_id)
                        else:
                            if no_input_name is not None:
                                search_button = driver.find_element(By.NAME, no_input_name)
                            else:
                                if no_input_class is not None:
                                    search_button = driver.find_element(By.CLASS_NAME, no_input_class[0])
                        search_button.click()
                        time.sleep(sleep_time)

        input_list = bs_obj.find_all('input')  # 获得所有input类型
        for a in input_list:
            if var_list[3] is not None and len(var_list[3]) > 0:
                for obj_index in var_list[3]:
                    obj_index_without = re_words(obj_index)
                    # for words_of_a in [a.get('name'), a.get('id')] + a.get('class'):
                    #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                    #         no_input_id = a.get('id')
                    #         no_input_class = a.get('class')
                    for x in [a.get('name'), a.get('id')]:
                        if x is not None and obj_index_without in x:
                            no_input_id = a.get('id')
                            no_input_class = a.get('class')
                            no_input_name = a.get('name')
                    if a.get('class') is not None:
                        for x in a.get('class'):
                            if x is not None and obj_index_without in x:
                                no_input_id = a.get('id')
                                no_input_class = a.get('class')
                                no_input_name = a.get('name')

                if no_input_id is None and no_input_name is None and no_input_class is None:
                    continue
                if no_input_id is not None:
                    search_button = driver.find_element(By.ID, no_input_id)
                else:
                    if no_input_name is not None:
                        search_button = driver.find_element(By.NAME, no_input_name)
                    else:
                        if no_input_class is not None:
                            search_button = driver.find_element(By.CLASS_NAME, no_input_class[0])
                search_button.click()
                time.sleep(sleep_time)
            else:
                if var_list[2] is not None and len(var_list[2]) > 0:
                    for obj_index in var_list[2]:
                        obj_index_without = re_words(obj_index)
                        # for words_of_a in [a.get('name'), a.get('id')] + a.get('class'):
                        #     if model.wv.similarity(obj_index_without, words_of_a) > 0.7:
                        #         no_input_id = a.get('id')
                        #         no_input_class = a.get('class')
                        for x in [a.get('name'), a.get('id')]:
                            if x is not None and obj_index_without in x:
                                no_input_id = a.get('id')
                                no_input_class = a.get('class')
                                no_input_name = a.get('name')
                        if a.get('class') is not None:
                            for x in a.get('class'):
                                if x is not None and obj_index_without in x:
                                    no_input_id = a.get('id')
                                    no_input_class = a.get('class')
                                    no_input_name = a.get('name')

                        if no_input_id is None and no_input_name is None and no_input_class is None:
                            continue
                        if no_input_id is not None:
                            search_button = driver.find_element(By.ID, no_input_id)
                        else:
                            if no_input_name is not None:
                                search_button = driver.find_element(By.NAME, no_input_name)
                            else:
                                if no_input_class is not None:
                                    search_button = driver.find_element(By.CLASS_NAME, no_input_class[0])
                        search_button.click()
                        time.sleep(sleep_time)
        return


def catch_for_five(step, driver, compare_text, number_of_bugs, text):  # 得到五元组
    # 初始化定义
    verb_count = 0  # 一共有几句
    now_of_count = 0  # 现在在处理第几个
    var_lists = [0]  # 初始没有断言 未实现
    verb_list = []  # 动作数组（句子里）
    have_http = False  # 是否有网址 特定动词区分是否跳转网页

    words = nlp(step)
    doc = words.sentences[0].words

    verb_count = count_verb(doc, verb_count, verb_list)
    has_assert(doc, var_lists)

    while now_of_count < verb_count:
        if now_of_count == verb_count - 1 and var_lists[0] == 1:
            var_list = [None] * 4  # 断言三元组初始化 ///
        else:
            var_list = [None] * 6  # 六元组初始化
        var_list[0] = verb_list[now_of_count]
        if ('http' or 'www') in step:
            have_http = True
        # 查找分词结果中的各种类型的单词。。每次只找当前动词后面词进行判断。找到第一个想要的词时break
        catch_prep(doc, now_of_count, var_list, verb_list)
        catch_obj(doc, now_of_count, var_list, verb_list, have_http)
        if var_lists[0] != 1 or now_of_count != verb_count - 1:
            catch_putin(doc, now_of_count, var_list, verb_list)
            if var_list[2] is not None:
                catch_adj_of_obl(doc, now_of_count, var_list, verb_list)
            if var_list[4] is not None:
                catch_adj_of_obj(doc, now_of_count, var_list, verb_list, text)

        now_of_count += 1  # 动词计数
        if open_driver == True:
            auto_step(var_list, driver)  # 生成脚本
        var_lists.append(var_list)
    print(var_lists)
    original_str = 'bug' + str(number_of_bugs)
    for index, list_of_var in enumerate(var_lists):
        if index != 0:
            for value in list_of_var:
                original_str += ' ' + str(value)
    # print_relation_of_step(words)  # 输出依赖关系
    if compare_text != original_str:  # 如果理解错误
        print('应该为：' + compare_text)
        print('实际为：' + original_str)
        print_relation_of_step(words)  # 输出依赖关系
        print('>>>>>>>>>>>>')
        print('>>>>>>>>>>>>')
        return 'fail'
    return 'success'


def analyze_bugs(number_of_bugs, all_count, failed_count, failed_list):  # spacy处理
    with open(r'./dom_bug/' + str(number_of_bugs) + '.txt', 'r',
              errors='ignore') as f, open(r'dom_bug/comparison.txt', 'r',
                                          errors='ignore') as f1:
        text = f.read().splitlines()
        compare = f1.read().splitlines()
        driver = webdriver.Chrome()

        steps = [elem for elem in text if 'STEP' in elem]

        # 合并步骤行下的内容行（两步骤之间属于第一个步骤的信息）
        steps_index = []
        for index, step in enumerate(steps):
            for i, sentence in enumerate(text):
                if sentence == step:
                    steps_index.append(i)

        now_index = 0
        while now_index < len(steps_index) - 1:
            now_now_index = steps_index[now_index] + 1
            while now_now_index < steps_index[now_index + 1]:
                if steps[now_index][len(steps[now_index]) - 1] == '.':
                    steps[now_index] = steps[now_index][0:len(steps[now_index]) - 1]
                steps[now_index] += " " + text[now_now_index]
                now_now_index += 1
            now_index += 1

        # 如果步骤信息第一步不含url则全局查找url
        find_url = re.compile('''(?:http://|www.|https://)[^\s]+''')
        if 'http' not in steps[0] and 'www.' not in steps[0]:
            for sentence in text:
                if 'login page:' in sentence:
                    login_url = find_url.findall(sentence)
                    driver.get(login_url[0])

        for index, step in enumerate(steps):  # 去除词之间多余的空格 enumerate带索引的便利
            new_str = ''
            for step_index, step_word in enumerate(step):
                if step_word == ' ' and step[step_index - 1] == ' ':
                    continue
                new_str += step_word
            steps[index] = new_str
        compare_res = [elem for elem in compare if 'bug' + str(number_of_bugs) in elem]  # 手写正确的数组 comparison.txt

        for index, step in enumerate(steps):
            print(step[6:len(step)], index)
            all_count += 1
            is_success = catch_for_five(step[6:len(step)], driver, compare_res[index], number_of_bugs, text)
            if is_success == 'fail':
                failed_list.append('bug' + str(number_of_bugs) + ' ' + 'step' + str(index + 1))
                failed_count += 1

        time.sleep(sleep_time)
        driver.close()

        f.close()
        print('第' + str(number_of_bugs) + '处理完毕')
        print('>>>>>>>>>>>>')
        print('>>>>>>>>>>>>')

        return [all_count, failed_count]


all_num = 0
failed_num = 0
failed_bug = []
for i in range(bug_begin, bug_end):  # 主函数，操作的bug文件
    [all_num, failed_num] = analyze_bugs(i, all_num, failed_num, failed_bug)
print('成功率为:' + str((all_num - failed_num) / all_num * 100) + '%')
print('失败位置：')
print(failed_bug)
