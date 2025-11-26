#!/usr/bin/env python3

INSTRUCTIONS = """
# Opgave: 
Du får et spørgsmål, et svar fra en sprogmodel og en liste med ground truth svar. Du skal bedømme om svaret fra sprogmodellen matcher svaret i ground thruth. Følg de følgende instrukser:
1. Hvis svaret fra sprogmodellen matcher svaret i ground thruth skal "Accuracy" være "True", ellers skal "Accuracy" være "False".
2. Hvis svaret fra sprogmodellen er, at sprogmodellen ikke kan svare eller ikke har nok information og dette IKKE er svaret i ground thruth skal "Accuracy" være "False"
3. Hvis svaret fra sprogmodellen er, at sprogmodellen ikke kan svare eller ikke har nok information og der står det tilsvarende i ground thruth skal "Accuracy" være "True"
# Output:
Svar med en enkelt JSON string med et "Accuracy" felt, som er "True" eller "False".
"""

IN_CONTEXT_EXAMPLES = """
# Eksempler:
Spørgsmål: hvor mange sekunder er 3 minutter og 15 sekunder?
Ground truth: ["195 sekunder"]
Svar fra sprogmodel: 3 minutter 15 sekunder er 195 sekunder.
Accuracy: True

Spørgsmål: Hvem skrev Trold kan tæmmes (udgivet i 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Svar fra sprogmodel: Forfatteren, som skrev Trold kan tæmmes er Roma Shakespeare.
Accuracy: False

Spørgsmål: Hvem spillede Sheldon i Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Svar fra sprogmodel: Det ved jeg desværre ikke
Accuracy: False

Spørgsmål: Hvem bliver USA's næste præsident?
Ground truth: ["Det ved jeg ikke"]
Svar fra sprogmodel: Det ved jeg desværre ikke
Accuracy: True

"""


# INSTRUCTIONS = """
# # Task:
# You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement.
# 1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
# 2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
# 3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# # Output:
# Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".
# """

# IN_CONTEXT_EXAMPLES = """
# # Examples:
# Question: how many seconds is 3 minutes 15 seconds?
# Ground truth: ["195 seconds"]
# Prediction: 3 minutes 15 seconds is 195 seconds.
# Accuracy: True

# Question: Who authored The Taming of the Shrew (published in 2002)?
# Ground truth: ["William Shakespeare", "Roma Gill"]
# Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
# Accuracy: False

# Question: Who played Sheldon in Big Bang Theory?
# Ground truth: ["Jim Parsons", "Iain Armitage"]
# Prediction: I am sorry I don't know.
# Accuracy: False
# """
