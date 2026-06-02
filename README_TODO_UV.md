F821 Undefined name `Compare_Retrievers`
  --> src/mitcfu_rag/evaluation_tools/compare_retrievers.py:21:18
   |
19 | def get_retrieval_models():
20 |     retrieval_model_instances = []
21 |     for Model in Compare_Retrievers:
   |                  ^^^^^^^^^^^^^^^^^^
22 |         retrieval_model_instances += [Model()]
23 |     return retrieval_model_instances
   |

E722 Do not use bare `except`
   --> src/mitcfu_rag/evaluation_tools/evaluation.py:109:5
    |
108 |         return answer
109 |     except:
    |     ^^^^^^
110 |         return -1
    |

E722 Do not use bare `except`
  --> src/mitcfu_rag/rag/langgraph_graphs.py:96:9
   |
94 |             json_response = json.loads(route_result)
95 |             agent = json_response.get("agent", None)
96 |         except:
   |         ^^^^^^
97 |             logger.info("Unable to parse response as json")
98 |             agent = None
   |

E722 Do not use bare `except`
   --> src/mitcfu_rag/rag/langgraph_graphs.py:139:9
    |
137 |             json_response = self._extract_json(reformulate_output)
138 |             reformulated_queries = json_response.get("søgninger", [])
139 |         except:
    |         ^^^^^^
140 |             logger.info("Unable to parse as json.")
141 |             reformulated_queries = []
    |

E402 Module level import not at top of file
  --> src/mitcfu_rag/rag/retrievers/dummy_retriever.py:27:1
   |
25 | ]
26 |
27 | import logging
   | ^^^^^^^^^^^^^^
28 | from mitcfu_rag.rag.rag import Retriever
   |

E402 Module level import not at top of file
  --> src/mitcfu_rag/rag/retrievers/dummy_retriever.py:28:1
   |
27 | import logging
28 | from mitcfu_rag.rag.rag import Retriever
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
29 |
30 | logger = logging.getLogger(__name__)
   |

F841 Local variable `query` is assigned to but never used
  --> src/mitcfu_rag/rag/retrievers/dummy_retriever.py:38:9
   |
37 |     def retrieve(self, messages: list[str]):
38 |         query = self.messages_to_query(messages)
   |         ^^^^^
39 |         similarities = [1.0, 0.0]
   |
help: Remove assignment to unused variable `query`

F841 Local variable `genre_and_form` is assigned to but never used
   --> src/mitcfu_rag/rag/retrievers/streaming_multilingual_retriever.py:117:9
    |
115 |             mat.get("general").get("specific") for mat in doc.get("materialTypes", [])
116 |         ]
117 |         genre_and_form = doc.get("genreAndForm", [])
    |         ^^^^^^^^^^^^^^
118 |         languages = [lan.get("display") for lan in doc.get("mainLanguages", [])]
119 |         creators_person = [
    |
help: Remove assignment to unused variable `genre_and_form`

F841 Local variable `new_prompt` is assigned to but never used
  --> src/mitcfu_rag/rag/summarizers/general_summarizer.py:72:9
   |
70 | """
71 |         )
72 |         new_prompt = (
   |         ^^^^^^^^^^
73 |             """
74 | Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
   |
help: Remove assignment to unused variable `new_prompt`

F841 Local variable `port` is assigned to but never used
   --> src/mitcfu_rag/term_ui.py:109:5
    |
107 |     """Commandline interface"""
108 |
109 |     port = 5000
    |     ^^^^
110 |     parser = argparse.ArgumentParser(description="term faktachat")
111 |     parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    |
help: Remove assignment to unused variable `port`

F403 `from .embedder import *` used; unable to detect undefined names
 --> src/mitcfu_rag/tools/__init__.py:3:1
  |
1 | __all__ = []
2 |
3 | from .embedder import *
  | ^^^^^^^^^^^^^^^^^^^^^^^
4 |
5 | __all__ += embedder.__all__
  |

F405 `embedder` may be undefined, or defined from star imports
 --> src/mitcfu_rag/tools/__init__.py:5:12
  |
3 | from .embedder import *
4 |
5 | __all__ += embedder.__all__
  |            ^^^^^^^^
6 |
7 | from .knn_searcher import *
  |

E402 Module level import not at top of file
 --> src/mitcfu_rag/tools/__init__.py:7:1
  |
5 | __all__ += embedder.__all__
6 |
7 | from .knn_searcher import *
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 |
9 | __all__ += knn_searcher.__all__
  |

F403 `from .knn_searcher import *` used; unable to detect undefined names
 --> src/mitcfu_rag/tools/__init__.py:7:1
  |
5 | __all__ += embedder.__all__
6 |
7 | from .knn_searcher import *
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 |
9 | __all__ += knn_searcher.__all__
  |

F405 `knn_searcher` may be undefined, or defined from star imports
  --> src/mitcfu_rag/tools/__init__.py:9:12
   |
 7 | from .knn_searcher import *
 8 |
 9 | __all__ += knn_searcher.__all__
   |            ^^^^^^^^^^^^
10 |
11 | from .semantic_splitter import *
   |

E402 Module level import not at top of file
  --> src/mitcfu_rag/tools/__init__.py:11:1
   |
 9 | __all__ += knn_searcher.__all__
10 |
11 | from .semantic_splitter import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | __all__ += semantic_splitter.__all__
   |

F403 `from .semantic_splitter import *` used; unable to detect undefined names
  --> src/mitcfu_rag/tools/__init__.py:11:1
   |
 9 | __all__ += knn_searcher.__all__
10 |
11 | from .semantic_splitter import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | __all__ += semantic_splitter.__all__
   |

F405 `semantic_splitter` may be undefined, or defined from star imports
  --> src/mitcfu_rag/tools/__init__.py:13:12
   |
11 | from .semantic_splitter import *
12 |
13 | __all__ += semantic_splitter.__all__
   |            ^^^^^^^^^^^^^^^^^
   |

Found 49 errors (31 fixed, 18 remaining).
No fixes available (4 hidden fixes can be enabled with the `--unsafe-fixes` option).
