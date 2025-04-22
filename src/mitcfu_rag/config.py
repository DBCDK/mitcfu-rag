#from fakta_chat.rag.dummy_rag import DummyRAG
#from fakta_chat.rag.solr_rag import SolrRAG
from mitcfu_rag.rag.streaming_rag import StreamingRAG
# the model that should be used in evaluation, chatUI

RAG = StreamingRAG

# for evaluation, the model that should be used for comparison
# flag -c skal sættes til True
#ComparisonRAG = SolrRAG


# # for comparing several retrieval models
# from fakta_chat.rag.retrievers.meta_solr_retriever import MetaSolrRetriever
# from fakta_chat.rag.retrievers.bm25_retriever import BM25Retriever
# from fakta_chat.rag.retrievers.mistrale5_instruct_retriever import Mistrale5Retriever
# from fakta_chat.rag.retrievers.ensemblers.reciprocal_rerank import ReciprocalEnsembler
# #from nily
# from fakta_chat.rag.retrievers.solr_retriever import SolrRetriever
# from fakta_chat.rag.retrievers.multilinguale5_large_retriever import EmbeddingRetriever

# Compare_Retrievers = [MetaSolrRetriever, BM25Retriever, Mistrale5Retriever, ReciprocalEnsembler, SolrRetriever, EmbeddingRetriever]
