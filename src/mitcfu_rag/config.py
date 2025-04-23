#from fakta_chat.rag.dummy_rag import DummyRAG
#from fakta_chat.rag.solr_rag import SolrRAG
from mitcfu_rag.rag.streaming_rag import StreamingRAG
from mitcfu_rag.rag.agent_streaming_rag import AgenticRAG

# the model that should be used in evaluation, chatUI
# MODELS TO USE
RAG = StreamingRAG
AGENTIC = AgenticRAG


# AGENT PROMPT TEMPLATES
RAG_TEMPLATE = {
    "name": "RAG",
    "description": "svarer på spørgsmål om en masse forskellige emner ved at bruge kilder fra faktalink.",
    "prompt": """
Du modtager et spørgsmål og nogle kilde. Din opgave er at besvare spørgsmål kun ved at bruge informationen i kilderne.
Det er ikke sikkert at nogen af kilderne er relevante for brugerens forespørgsel.
Du overholder følgende regler:
- Du svarer aldrig på spørgsmål, hvor du ikke kan finde svaret i kilderne.
- Du opfinder aldrig kilder.
- Du skriver aldrig links til websider.
- Du svarer altid på dansk.
- Hvis ikke du kan finde svaret, forklarer du at du ikke kan finde svaret, og beder dem omformulere spørgsmålet.
- Dit output er kun dit svar, ikke kilderne på dit svar.

Kilder:
"""
}

SIMPLE_TEMPLATE = {
    "name": "SIMPLE",
    "description": "svarer KUN på simple ting som hej, tak, og forklaring på hvad faktachat er.",
    "prompt":"""
Brugeren har stillet et spørgsmål der ikke handler om specifikke faktalink artikler, eller sagt hej, tak eller farvel.
Du svarer høftligt og kortfattet brugeren med en afslappet tone.
Hvis spørgsmålet ikke er noget i stil med "hej" eller "tak", så forklarer du brugeren at du er Faktachat
og beder dem stille et spørgsmål som du kan hjælpe med at svare på.
Chat-historik:
"""}

FALLBACK_TEMPLATE = {
    "name": "FALLBACK",
    "description": "hvis spørgsmålet falder uden for alle andre agenter hjælper denne her brugeren på rette spor igen",
    "prompt": """
Brugeren spørger om noget der ikke er relevant for Faktachat. Forklar brugeren at du ikke kan besvare deres spørgsmål,
og bed dem om at spørge om noget andet.    
"""
}

def ROUTER_TEMPLATE():
    ALL_TEMPLATES = [SIMPLE_TEMPLATE, FALLBACK_TEMPLATE, RAG_TEMPLATE]
    return {
        "name": "ROUTER",
        "descrption": "vælger hvilken agent der skal svare på den seneste besked.",
        "prompt":"""
    Brugeren har sendt en besked, og det er din opgave at bedømme hvilken agent der skal håndtere beskeden.
    Du skal kigge på den seneste besked og bedømme hvilken agent der skal besvare det.
    """ + "\n".join([f"[{TEMP['name']}] : {TEMP['description']}\n" for TEMP in ALL_TEMPLATES]) + """
    Dit svar skal være KUN, IKKE ANDET end """ + "eller ".join([f"[{TEMP['name']}]" for TEMP in ALL_TEMPLATES]) + """\n\n
    """}


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
