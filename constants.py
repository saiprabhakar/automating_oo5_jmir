option_prompt_original = "We want to know if the doctor has communicated all the treatement and decision options to the patient, so that the patient can make an informed medical decision. As a medical reviewer, your task is to carefully review the above conversation and identify the most relevant instances (top 5) where the doctor explain/communicates to the patient about the options or choices in their medical treatment and decisions they have make. Additionally, include utterances where the doctor explains the details of each of the options the patient has in detail. Do not include sentences that are not relevant for the patient in their medical decision making. Do not include sentence where the doctor is collecting information from the patient. Please provide specific sentence IDs for each instance you find. Dont produce any sentence ids if you didnt find anything. Explain your decision for each extraction briefly. Write the individual sentence ids as <sentence id: sentence id >, and span of sentences as <sentence ids: start sentence - end sentence >"

# option_prompt += ". Make sure the sentence ids are present in the conversation."

option_prompt_no_explanation = "We want to know if the doctor has communicated all the treatement and decision options to the patient, so that the patient can make an informed medical decision. As a medical reviewer, your task is to carefully review the above conversation and identify the most relevant instances (top 5) where the doctor explain/communicates to the patient about the options or choices in their medical treatment and decisions they have make. Additionally, include utterances where the doctor explains the details of each of the options the patient has in detail. Do not include sentences that are not relevant for the patient in their medical decision making. Do not include sentence where the doctor is collecting information from the patient. Please provide specific sentence IDs for each instance you find. Dont produce any sentence ids if you didnt find anything. Write the individual sentence ids as <sentence id: sentence id >, and span of sentences as <sentence ids: start sentence - end sentence >"

system_prompt = "You are a medical reviewer."

example_formatted_inputs3 = [
"""88 - There are two options for the treatment
...
97 - I can tell you my opinions, but the decision is yours to make.
""",
"""88 - Ofcourse I will explain your options and compare them for you.
...
97 - In option A we will do ...
98 - In option B we will do ....
...
109 - The decision will be yours to make.
...
116 - Is the risk low in both cases?
117 - In both cases the risk is similar and low.
...
125 - The main differences between the two options are ...
"""
]

example_formatted_outputs3 = [
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 88> - The doctor says there are two options for the treatment.
<sentence id: 97> - Doctor is going to tell the patient what they think about the options. And the doctor says that the decision is patient’s to make.
""",
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 88> - The doctor says they will explain the options for the patient.
<sentence ids: 97 - 98> - Doctor explains the two options with an aim of comparing them.
<sentence id: 109> - Doctor points out that the decision is patient's to make.
<sentence ids: 116 - 117> - The doctor answers the patient's question in comparing the options they have. In addition they also explain the differeces.
<sentence id: 125> - Doctor points out the differnce between the two options they have.
"""
]

example_formatted_outputs3_no_explanation = [
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 88>
<sentence id: 97>
""",
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 88> 
<sentence ids: 97 - 98>
<sentence id: 109>
<sentence ids: 116 - 117>
<sentence id: 125>
"""
]

example_formatted_inputs4 = [
"""8 - There are two options for the treatment
...
10 - I can tell you my opinions, but the decision is yours to make.
""",
"""11 - Ofcourse I will explain your options and compare them for you.
...
19 - In option A we will do ...
20 - In option B we will do ....
...
25 - The decision will be yours to make.
...
29 - Is the risk low in both cases?
30 - In both cases the risk is similar and low.
...
35 - The main differences between the two options are ...
"""
]

example_formatted_outputs4 = [
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 8> - The doctor says there are two options for the treatment.
<sentence id: 10> - Doctor is going to tell the patient what they think about the options. And the doctor says that the decision is patient’s to make.
""",
"""Instances where the doctor explains/communicates treatment and decision options to the patient:
<sentence id: 11> - The doctor says they will explain the options for the patient.
<sentence ids: 19 - 20> - Doctor explains the two options with an aim of comparing them.
<sentence id: 25> - Doctor points out that the decision is patient's to make.
<sentence ids: 29 - 30> - The doctor answers the patient's question in comparing the options they have. In addition they also explain the differeces.
<sentence id: 35> - Doctor points out the differnce between the two options they have.
"""
]

example_inputs = example_formatted_inputs3
example_outputs = example_formatted_outputs3_no_explanation

BATCH_SIZE = 80
PROIMITY_THRESHOLD = 2
OFFSET = 1
USE_ZERO_ALL_BATCHES = False