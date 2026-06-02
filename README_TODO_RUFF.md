F821 Undefined name `Compare_Retrievers`
  --> src/science_rag/evaluation_tools/compare_retrievers.py:21:18
   |
19 | def get_retrieval_models():
20 |     retrieval_model_instances = []
21 |     for Model in Compare_Retrievers:
   |                  ^^^^^^^^^^^^^^^^^^
22 |         retrieval_model_instances += [Model()]
23 |     return retrieval_model_instances
   |

E722 Do not use bare `except`
   --> src/science_rag/evaluation_tools/evaluation.py:109:5
    |
108 |         return answer
109 |     except:
    |     ^^^^^^
110 |         return -1
    |

F841 Local variable `processed_messages` is assigned to but never used
  --> src/science_rag/rag/demo_rag.py:53:9
   |
51 |         Revieves a list of chat messages and returns the next response given by the chatbot.
52 |         """
53 |         processed_messages = self.parser(messages)
   |         ^^^^^^^^^^^^^^^^^^
54 |         similarities, references = self.retriever(messages, n=3)
   |
help: Remove assignment to unused variable `processed_messages`

F841 Local variable `processed_messages` is assigned to but never used
  --> src/science_rag/rag/embedding_with_history_rag.py:54:9
   |
52 |         """
53 |         # print(messages)
54 |         processed_messages = self.parser(messages)
   |         ^^^^^^^^^^^^^^^^^^
55 |         similarities, references = self.retriever(messages, n=3)
   |
help: Remove assignment to unused variable `processed_messages`

F841 Local variable `endpoint_url` is assigned to but never used
   --> src/science_rag/rag/generators/agent_streaming_generator.py:251:9
    |
249 |         endpoint_profile = input.get("endpoint_profile", "tgi")
250 |         endpoint_lookup = self.vllm_endpoints if endpoint_profile == "vllm" else self.tgi_endpoints
251 |         endpoint_url = endpoint_lookup.get(input["model_name"], self.tgi_endpoints[input["model_name"]])
    |         ^^^^^^^^^^^^
252 |
253 |         endpoint_profile = input.get("endpoint_profile", "tgi")
    |
help: Remove assignment to unused variable `endpoint_url`

E722 Do not use bare `except`
  --> src/science_rag/rag/langgraph_graphs.py:94:9
   |
92 |             json_response = json.loads(route_result)
93 |             agent = json_response.get("agent", None)
94 |         except:
   |         ^^^^^^
95 |             logger.info("Unable to parse response as json")
96 |             agent = None
   |

E722 Do not use bare `except`
   --> src/science_rag/rag/langgraph_graphs.py:131:9
    |
129 |             json_response = self._extract_json(reformulate_output)
130 |             reformulated_queries = json_response.get("søgninger", [])
131 |         except:
    |         ^^^^^^
132 |             logger.info("Unable to parse as json.")
133 |             reformulated_queries = []
    |

E402 Module level import not at top of file
  --> src/science_rag/rag/retrievers/dummy_retriever.py:27:1
   |
25 | ]
26 |
27 | import logging
   | ^^^^^^^^^^^^^^
28 | from mitcfu_rag.rag.rag import Retriever
   |

E402 Module level import not at top of file
  --> src/science_rag/rag/retrievers/dummy_retriever.py:28:1
   |
27 | import logging
28 | from mitcfu_rag.rag.rag import Retriever
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
29 |
30 | logger = logging.getLogger(__name__)
   |

F841 Local variable `query` is assigned to but never used
  --> src/science_rag/rag/retrievers/dummy_retriever.py:38:9
   |
37 |     def retrieve(self, messages: list[str]):
38 |         query = self.messages_to_query(messages)
   |         ^^^^^
39 |         similarities = [1.0, 0.0]
   |
help: Remove assignment to unused variable `query`

E402 Module level import not at top of file
   --> src/science_rag/rag/retrievers/solr_retriever.py:330:1
    |
328 | ]
329 |
330 | import logging
    | ^^^^^^^^^^^^^^
331 | from mitcfu_rag.rag.rag import Retriever, Reference
332 | import dbc_pyutils.solr
    |

E402 Module level import not at top of file
   --> src/science_rag/rag/retrievers/solr_retriever.py:331:1
    |
330 | import logging
331 | from mitcfu_rag.rag.rag import Retriever, Reference
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
332 | import dbc_pyutils.solr
333 | from keybert import KeyBERT
    |

E402 Module level import not at top of file
   --> src/science_rag/rag/retrievers/solr_retriever.py:332:1
    |
330 | import logging
331 | from mitcfu_rag.rag.rag import Retriever, Reference
332 | import dbc_pyutils.solr
    | ^^^^^^^^^^^^^^^^^^^^^^^
333 | from keybert import KeyBERT
    |

E402 Module level import not at top of file
   --> src/science_rag/rag/retrievers/solr_retriever.py:333:1
    |
331 | from mitcfu_rag.rag.rag import Retriever, Reference
332 | import dbc_pyutils.solr
333 | from keybert import KeyBERT
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
334 |
335 | logger = logging.getLogger(__name__)
    |

F841 Local variable `processed_messages` is assigned to but never used
  --> src/science_rag/rag/solr_rag.py:60:9
   |
58 |         """
59 |         print(messages)
60 |         processed_messages = self.parser(messages)
   |         ^^^^^^^^^^^^^^^^^^
61 |         similarities, references = self.retriever(messages)
   |
help: Remove assignment to unused variable `processed_messages`

F841 Local variable `processed_messages` is assigned to but never used
  --> src/science_rag/rag/solr_rag.py:97:9
   |
96 |         messages = [{"role": "user", "content": messages[0]}]
97 |         processed_messages = self.parser(messages)
   |         ^^^^^^^^^^^^^^^^^^
98 |         similarities, references = self.retriever(messages)
   |
help: Remove assignment to unused variable `processed_messages`

F841 Local variable `new_prompt` is assigned to but never used
  --> src/science_rag/rag/summarizers/general_summarizer.py:70:9
   |
68 | """
69 |         )
70 |         new_prompt = (
   |         ^^^^^^^^^^
71 |             """
72 | Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
   |
help: Remove assignment to unused variable `new_prompt`

E402 Module level import not at top of file
 --> src/science_rag/streamlit_ui.py:5:1
  |
3 | torch.classes.__path__ = []  # type: ignore
4 |
5 | import os
  | ^^^^^^^^^
6 | import random
7 | import json
  |

E402 Module level import not at top of file
 --> src/science_rag/streamlit_ui.py:6:1
  |
5 | import os
6 | import random
  | ^^^^^^^^^^^^^
7 | import json
8 | import streamlit as st
  |

E402 Module level import not at top of file
 --> src/science_rag/streamlit_ui.py:7:1
  |
5 | import os
6 | import random
7 | import json
  | ^^^^^^^^^^^
8 | import streamlit as st
9 | import requests
  |

E402 Module level import not at top of file
 --> src/science_rag/streamlit_ui.py:8:1
  |
6 | import random
7 | import json
8 | import streamlit as st
  | ^^^^^^^^^^^^^^^^^^^^^^
9 | import requests
  |

E402 Module level import not at top of file
  --> src/science_rag/streamlit_ui.py:9:1
   |
 7 | import json
 8 | import streamlit as st
 9 | import requests
   | ^^^^^^^^^^^^^^^
10 |
11 | from science_rag.tools.llm_formatting import select_model_function, GEMMA_3_12B
   |

E402 Module level import not at top of file
  --> src/science_rag/streamlit_ui.py:11:1
   |
 9 | import requests
10 |
11 | from science_rag.tools.llm_formatting import select_model_function, GEMMA_3_12B
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | # from fakta_chat.config import RAG
   |

F841 Local variable `port` is assigned to but never used
   --> src/science_rag/term_ui.py:109:5
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
 --> src/science_rag/tools/__init__.py:3:1
  |
1 | __all__ = []
2 |
3 | from .embedder import *
  | ^^^^^^^^^^^^^^^^^^^^^^^
4 |
5 | __all__ += embedder.__all__
  |

F405 `embedder` may be undefined, or defined from star imports
 --> src/science_rag/tools/__init__.py:5:12
  |
3 | from .embedder import *
4 |
5 | __all__ += embedder.__all__
  |            ^^^^^^^^
6 |
7 | from .knn_searcher import *
  |

E402 Module level import not at top of file
 --> src/science_rag/tools/__init__.py:7:1
  |
5 | __all__ += embedder.__all__
6 |
7 | from .knn_searcher import *
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 |
9 | __all__ += knn_searcher.__all__
  |

F403 `from .knn_searcher import *` used; unable to detect undefined names
 --> src/science_rag/tools/__init__.py:7:1
  |
5 | __all__ += embedder.__all__
6 |
7 | from .knn_searcher import *
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 |
9 | __all__ += knn_searcher.__all__
  |

F405 `knn_searcher` may be undefined, or defined from star imports
  --> src/science_rag/tools/__init__.py:9:12
   |
 7 | from .knn_searcher import *
 8 |
 9 | __all__ += knn_searcher.__all__
   |            ^^^^^^^^^^^^
10 |
11 | from .semantic_splitter import *
   |

E402 Module level import not at top of file
  --> src/science_rag/tools/__init__.py:11:1
   |
 9 | __all__ += knn_searcher.__all__
10 |
11 | from .semantic_splitter import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | __all__ += semantic_splitter.__all__
   |

F403 `from .semantic_splitter import *` used; unable to detect undefined names
  --> src/science_rag/tools/__init__.py:11:1
   |
 9 | __all__ += knn_searcher.__all__
10 |
11 | from .semantic_splitter import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | __all__ += semantic_splitter.__all__
   |

F405 `semantic_splitter` may be undefined, or defined from star imports
  --> src/science_rag/tools/__init__.py:13:12
   |
11 | from .semantic_splitter import *
12 |
13 | __all__ += semantic_splitter.__all__
   |            ^^^^^^^^^^^^^^^^^
   |

Found 93 errors (61 fixed, 32 remaining).
No fixes available (8 hidden fixes can be enabled with the `--unsafe-fixes` option).
