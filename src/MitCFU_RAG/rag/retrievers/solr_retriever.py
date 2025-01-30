#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
:mod:`fakta_chat.solr_retriever - solr_retriever

============
SolrRetriever
============

SolrRetriever retrieves relevant references based on the messages from the chat sent.

First performs two searches:
1. extracts keywords using keyBERT, and performs a keyword only search, ignoring words in stop_words list
2. performs a general search on the full question

Then combines the result of the two searches, removing duplicates. Simple mix, just taking first from keyword, then first from general, etc.

#TODO use the solr functions from https://gitlab.dbc.dk/ai/fakta-chat-solr

example of usage:

    d_retriever = SolrRetriever()
    messages = messages = ["Hej", "Er der noget om biblioteker?"]
    refs = d_retriever.retrieve(messages)
    print(f'relevant references: {refs}')
"""

STOP_WORDS = ["ad","af","al","aldrig","alene","alle","allerede","alligevel","alt","altid","andet","andre","at","bag",
"bare","begge","bile","blandt","blev","blive","bliver","blot","bruge","burde","bør","både","da","de","dem","den","denne","dens",
"der","derefter","deres","derfor","derfra","deri","dermed","derpå","derved","det","dette","dig","din","dine","disse","dit","dog",
"du","efter","egen","ej","eller","ellers","en","end","endnu","ene","eneste","enhver","ens","enten","er","et","far","fem","fik",
"fire","flere","flest","fleste","for","foran","fordi","forrige","fra","få","får","før","først","gang","gennem","gerne","gik","giver",
"gjorde","gjort","god","godt","gør","gøre","gørende","går","ham","han","handler","har","havde","have","hej","hel","hele","heller","helt","hen",
"hende","hendes","henover","her","herefter","heri","hermed","herpå","hos","hun","hvad","hvem","hver","hvilke","hvilken","hvilkes","hvis",
"hvor","hvordan","hvorefter","hvorfor","hvorfra","hvorhen","hvori","hvorimod","hvornår","hvorved","i","ifølge","igen","igennem","ikke",
"imellem","imens","imod","in","ind","indtil","ingen","intet","ja","jeg","jer","jeres","jo","kan","kom","komme","kommer","kun","kunne",
"lad","langs","le","lav","lave","lavet","lidt","lige","ligesom","lille","lo","længere","made","man","mand","mange","mangen","med","megen","meget",
"mellem","men","mene","mener","mens","mere","mest","mig","min","mindre","mindst","mine","mit","mod","må","måde","måske","måtte","ned","nej",
"nemlig","netop","ni","nogen","nogensinde","noget","nogle","nok","nu","når","nær","næste","næsten","of","og","også","okay","om","omkring","op",
"oppe","os","ord","otte","over","overalt","par","på","sagde","samme","sammen","se","seks","selv","selvom","senere","ser","ses","side","sidder",
"siden","sidste","sig","sige","sin","sine","sit","skal","skulle","som","stadig","stod","stor","store","står","synes","syntes","syv","så","sådan","således",
"tag","tage","temmelig","the","thi","ti","tidligere","til","tilbage","ting","tit","to","tre","ud","uden","udover","under","undtagen","var","ved",
"vej","vi","via","vil","ville","vor","vore","vores","vær","være","været","with","www","you","øvrigt","åre", "srkive", "opgave", "skrive", "fungerer"]

import logging
from fakta_chat.rag.rag import Retriever, Reference
import dbc_pyutils.solr
from keybert import KeyBERT

logger = logging.getLogger(__name__)


class SolrRetriever(Retriever):
    def __init__(self):
        self.solr_url = "http://xpdev-p01:8800/solr/fakta-chat-solr/"
        self.solr = dbc_pyutils.solr.Solr(self.solr_url) if len(self.solr_url) > 0 else None
        self.keybert_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")

    def retrieve(self, messages: list[str], n: int = 5):
        #print(messages)
        query = messages[-1]["content"]
        #print("message: ", query)
        keywords = self.extract_keywords(query)
        #print("Keywords: ", keywords)
        keyword_query = " ".join([k for k, v in keywords])
        #print("k_query:", keyword_query)
        formatted_query = self.format_keyword_query(keyword_query)
        solr_keyword_response = self.solr.query(formatted_query)
        formatted_question_query = self.format_question_query(query)
        solr_question_response = self.solr.query(formatted_question_query)
        #print("querstion response: ", [doc["article_headline"] for doc in solr_question_response])
        #print("keyword response: ", [doc["article_headline"] for doc in solr_keyword_response])

        combined_solr_responses = self.combine_solr_responses(solr_keyword_response, solr_question_response)
    
        references = []
        seen_articles = set()
        for doc in combined_solr_responses:
            if doc["id"] in seen_articles:
                continue
            seen_articles.add(doc["id"])
            if "sentences" in doc:
                sentence = " ".join(doc["sentences"])
                #references.append({"id": doc["id"], "sentences": sentence.strip(), "article_link": doc["article_link"].strip(),
                #                   "article_headline": doc["article_headline"].strip(), "subheadline": doc["subheadline"].strip()})
                references.append(Reference(id=doc["id"],
                                            article_headline=doc["article_headline"].strip(),
                                            article_link=doc["article_link"],
                                            text=sentence.strip()))
        return [i for i, _ in enumerate(solr_keyword_response)][:n], references[:n]
    
    def extract_keywords(self, messages):
        #print("\n\nKEYBERT DEBUG")
        #print("QUERY: ", messages)
        keywords = self.keybert_model.extract_keywords(messages, keyphrase_ngram_range=(1, 2), stop_words=STOP_WORDS)
        #print("KEYWORDS: ", keywords)
        #print("KEYBERT DEBUG END\n\n")
        return keywords
    
    def format_question_query(self, input_string):
        return {
            "query": f'+all:("{input_string}" {input_string})',
            "fields": "id sentences article_link article_headline subheadline",
            "offset": 0,
            "limit": 10,
            "params": {
                "defType": "edismax",
                "f.all.qf": "article_headline subheadline^2 sentences",
                "sort": "score desc"
            }
        }
    
    def format_keyword_query(self, input_string):
        return {
            "query": f'+all:("{input_string}" {input_string})',
            "fields": "id sentences article_link article_headline subheadline",
            "offset": 0,
            "limit": 10,
            "params": {
                "defType": "edismax",
                "f.all.qf": "article_headline^2 article_topics^2 sentences^2 gen_keywords gen_questions",
                "sort": "score desc"
            }
        }
    
    def combine_solr_responses(self, keyword_response, question_response):
        """
        Takes two solr responses and merges them intertwined with no duplicates
        """
        question_response = [q for q in question_response]


        keyword_response = [q for q in keyword_response]

        seen_ids = set()
        combined_response = []
        for i in range(max(len(keyword_response), len(question_response))):
            if i < len(keyword_response):
                if not keyword_response[i]["id"] in seen_ids:
                    combined_response.append(keyword_response[i])
                    seen_ids.add(keyword_response[i]["id"])
            if i < len(question_response):
                if not question_response[i]["id"] in seen_ids:
                    combined_response.append(question_response[i])
                    seen_ids.add(question_response[i]["id"])

        return combined_response

