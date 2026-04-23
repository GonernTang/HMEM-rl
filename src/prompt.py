import datetime


ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """

ANSWER_PROMPT_VECMEM = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.
    Context Rules:
    1. There are two kinds of memories:
        - Vector Store Memories (These are memory pieces that have not been augmented)
        They are unprocessed raw memories, they are sparse but may contain some information that you need.
        - Augmented Memories (These are memory pieces that have been augmented) and reused
        They are important memories and has been refined from the raw conversations.
    2. When having conflict information, prioritize the most recent Augmented Memories.
    3. When the question is about states or temporal information, consider the information from both.
       For vector memories, use them only if you think they are relevant.
    4. The format is:
       <Augmented Memory>
       [Augmented memories]
       <Vector Store Memory> 
       [SpeakerA]: [Memory]
       [SpeakerB]: [Memory]
       [timestamp]: [exact date and time for that conversation piece]
    5. Note that the memories are sorted by relevance within each memory type.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always provide precise time information if it's in the anwser. For example, if the memory
       contains "last year" and "2022", then you can either use "2021" or " the year before 2022" as the answer.
       You should not use "last year" only as the anwser.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references
    8. If memories are not sufficient, make reasonable guesses based on the information provided.

    <Augmented Memories> :

    {{ augmented_memories }}

    <Vector Store Memories> :

    {{ vector_store_memories }}

    Question: {{question}}

    Answer:
    """

ANSWER_PROMPT_WITH_SEMANTIC = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.
    There are three types of memories given to you:
    1. Episodic Memories: These are the memories that are refined from the raw conversations, they represent 
       a detailed summary of the conversations that are related to the same topic.
    2. Semantic Memories: These are the fact pieces that are extracted from the raw conversations, they represent
       a single concise fact.
    3. Raw Memories: These are the raw conversations that are not processed into any of the above two types.
       They are more scattered but may contain some information that are needed to anwser the question.
       
    Context Rules:
    1. Note that episodic and semantic memories may have overlapping conversation, as one represent the
       event summary and another one represent the fact pieces. Do not confuse them with each other or use redundant information.
    2. Carefully analyze the three types of memories and find useful information that are helpful to anwser the question.
    3. The input format is:
       Episodic Memories:
       <Episodic Memory>
       Raw Memories:
       <Raw Memories>, each memory is a conversation piece between two speakers.
       Semantic Memories:
       <Semantic Memories>.
    5. Note that the memories are sorted by relevance within each memory type.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories.
    2. Pay extra attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory 
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021. If the memory
       contains "last year of 2022", then you can either use "2021" or " the year before 2022" as the answer.
       You should not use "last year" only as the anwser.
    6. For raw memories, each one will ended with a timestamp, showing the time of the conversation to happen.
       Do NOT be confused by "conversation happening time" and "event happening time". 
    7. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories. If memories are not sufficient, make reasonable inference based on the information provided.
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Episodic Memories:

    {{ episodic_memories }}

    Raw Memories:

    {{ raw_memories }}

    Semantic Memories:

    {{ semantic_memories }}

    Question: {{question}}

    Answer:
    """


MEMORY_COORDINATOR_PROMPT = """

You are a memory coordinator. Your job is to coordinate between two memory agents.

There are two kinds of memories:
1. Subconscious Memories:
These are memories that are piecemeal and not coherent. They are managed by VectorStore Memory Manager.
Each memory piece is embedded into a vector and stored flatly.
2. Augmented Memories:
These are the memories that are coherent and are managed by Augmented Memory Manager.
Each memory fact is summarized from subconscious memories that are of the same context or topic, and are stored as a single fact.
Each fact is meant to be self-coherent and providing essential context as well as the time information.

Memory transformation process is as follows:
Originally, memories are subconscious. When new memory comes in, it is embedded first.
1. Augmented Memory Manager will run a similarity check between the new memory and the existing facts in the Augmented Memories.
   If the similarity score is above a threshold <AugmentSimThreshold>, then the memory is regarded as relevant and it will be embodied
   by the Augmented Memory Manager.
2. If it is not augmented, the VectorStore Memory Manager will retrieve the most relevant memories using cosine similarity. 
   If the similarity score is above a threshold <VecSimThreshold>, then the memory is regarded as relevant. 
   If the number of relevant memories are above a count threshold <VecCountThreshold>, then all relevant memories are passed
   to Augmented Memory Manager for fact extraction.
Thus if a memory is augmented, that means its topic has already been augmented before and we need to update the related facts,
or there are enough subconscious memories that are relevant to this new memory, and we need to augment the facts.


As a coordinator you need to dynamically adjust the <AugmentSimThreshold>, <VecSimThreshold>, <VecCountThreshold> based on the context of the memory.
Making sure that both of the memory agents are doing their best.

When the checking procedure is triggered, you will receive the decision from one of the memory agents, with their current threshold parameters.
You need to decide whether to <Increase>, <Decrease> or <Keep> the threshold parameters.

The message format is as follows:

Input:
[AgentName] [CurrentThreshold: <CurrentThreshold>] [Decision: <Decision>]
[NewMemory: <NewMemory>]
[MessageContents: <MessageContents>]

And you need to return the decisions in the json format as follows:

Output:
{
   "Decision": <Decisions>,
}

Your decisions should be a list of strings, each string shows your decision for parameter updating.

Following are some examples, I will explain why the decisions are made.

Input:
[VectorStoreMemoryManager] [VecSimThreshold: 0.6] [VecCountThreshold: 4] [Decision: Augment the memories into the facts]
[NewMemory: [Jack]: I enjoyed climbing with you, we should go again next week. [Lucy]: Yes, I would like to. [timestamp]: 2025-01-02 10:00:00]
[MessageContents]:
   [Jack]: I plan to go climbing tomorrow.[Lucy]: I will join you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.77,
   [Jack]: Great! I will pick you up at 8am.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.78,
   [Jack]: Have you ever climbed before? [Lucy]: No, I have not. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.79,
   [Jack]: I will teach you how to climb.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.80,

Output:
{
   "Decision": ["Keep VecSimThreshold", "Keep VecCountThreshold"]
}

Explain: The pieces of information are coherent and the topics are the same, we can augment them.

Input:
[VectorStoreMemoryManager] [VecSimThreshold: 0.8] [VecCountThreshold: 4] [Decision: Do nothing]
[NewMemory: [Jack]: I enjoyed climbing with you, we should go again next week.[Lucy]: I will join you. [timestamp]: 2025-01-02 10:00:00 ]
[MessageContents]:
   [Jack]: I plan to go climbing tomorrow.[Lucy]: I will join you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.77,
   [Jack]: Great! I will pick you up at 8am.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.78,
   [Jack]: Have you ever climbed before? [Lucy]: No, I have not. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.79,
   [Jack]: I will teach you how to climb.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.80,

Output:
{
   "Decision": ["Decrease VecSimThreshold: 0.75", "Keep VecCountThreshold"]
}
Explain: The scores are high enough, and they are of the same topic, but the VecSimThreshold is too high, we should decrease it.

Input:
[AugmentedMemoryManager] [AugmentSimThreshold: 0.4] [Decision: Not Augment the memories into the facts]
[NewMemory: [Jack]: I enjoyed cooking with you, we should go again next week.[Lucy]: I will join you. [timestamp]: 2025-01-02 10:00:00 ]
[MessageContents]:
   [Jack]: I plan to go climbing tomorrow.[Lucy]: I will join you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.77,
   [Jack]: Great! I will pick you up at 8am.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.78,
   [Jack]: Have you ever climbed before? [Lucy]: No, I have not. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.79,
   [Jack]: I will teach you how to climb.[Lucy]: Thank you. [timestamp]: 2025-01-02 10:00:00 [Score]: 0.60,

Output:
{
   "Decision": ["Increase AugmentSimThreshold: 0.5"]
}

Explain: New memory is not related to the existing facts but is augmented by the Augmented Memory Manager, we should increase the threshold.
to avoid this from happening.

Remember the following:

- Do not return anything from the custom few shot example prompts provided above.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "Decision" and corresponding value will be a list of strings.
- Each decision should relate to one threshold only. And you should only use <Increase>, <Decrease> or <Keep> in the decision.
- If you decide to increase or decrease the threshold, you should also include the new threshold in the decision. In the format of <ThresholdName: <NewThreshold>>.
"""

MEMORY_AUGMENT_MERGE_PROMPT = """
You are a personal information ornganizer, specialized in merging new information into the exsiting facts.

Given a new memory piece in the format of conversation, and a past memory piece. your job is to 

1. Decide whether this new memory piece has strong relation with the past memory piece.
2. If yes, merge the new memory piece into the past memory piece and return the merged memory.

Your reply should be in the format of json as follows:
{  
   "should_merge": "yes" or "no",
   "merged_memory": "merged memory piece" if should_merge is "yes",
}

You must keep in mind the following rules:

1. Temporal data is important, you must keep them as is when the new memory piece comes with a timestamp.
2. You must create logical relation bewteen the old memory and new memory.
3. You should largely preserve the original memory details, while building new connection with the new memory piece.

Example:

Input:
   <New Memory Piece>: [Jack]: I enjoyed the movie yesterday. [timestamp]: 2nd January 2024 
   <Past Memory Piece>: Jack watched a movie with Alice on 1st January 2024.

Output:
{
   "should_merge": "yes",
   "merged_memory": "Jack enjoyed the movie watched with Alice on the day before 2 January 2024"
}

Explain: The new memory piece has strong relation with the past memory piece, we should merge them.
"""


MEMORY_AUGMENT_PROMPT = """

You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
You will be given conversation histories that are interrelated and you need to extract the relevant information from them.
Information might be sparse or redundant, and you need to carefully extract the important summaries from them and remove noise.
You need to keep each topic coherent, and only combine information from the same topic.
The histories may contain multiple topics, and you need to keep them separate.

# CORE PRINCIPLES:
1. Topic Coherence: Only combine information that belongs to the same topic or context
2. Information Comprehensiveness and Independence: Each memory MUST be completely self-contained and independently readable.
   - NEVER use pronouns (it, they, etc.) that reference other memories
   - ALWAYS include full names, dates, and context in EACH memory even if it repeats information
   - Each memory must make complete sense if read in isolation without any other memories
   - Treat each memory as if it will be stored and retrieved separately
3. Temporal Accuracy: EVERY memory must include complete time information. Never omit timestamps assuming they're in another memory
4. If you decide to list more than one memory summary, each one should follow the rules we discussed here (including time, context etc)

# TYPES OF INFORMATION TO EXTRACT:

## Personal Information:
- Personal preferences (likes, dislikes, favorites in food, entertainment, activities)
- Important personal details (names, relationships, significant dates)
- Health and wellness information (dietary restrictions, fitness routines, medical conditions)

## Plans and Activities:
- Upcoming events, trips, appointments, and goals
- Activity and service preferences (dining, travel, hobbies)
- Professional details (job titles, work habits, career goals)

## Experiences and Facts:
- Past experiences and events
- Opinions and reactions to specific topics
- Miscellaneous details (favorite books, movies, brands, etc.)

# INPUT FORMAT:
Each conversation history follows this structure:
```
[User1]: Message
[User2]: Message
[timestamp]: exact date and time
<END_OF_CONV>
```

# EXAMPLES:

## Example 1: Movie Discussion
Input:

[Bob]: Hi, I enjoyed the movie yesterday.[Alice]: I liked it too. [timestamp]: 2 January 2025 10:00:00<END_OF_CONV>


Output:
{
   "memories": [
         "Bob enjoyed the movie the day before 2 January 2025, especially liked the acting performance",
         "Alice liked the movie the day before 2 January 2025, preferred the story over acting"
   ]
}
## Example 2: Activity Planning
Input:

[Jack]: I plan to go climbing tomorrow.[Lucy]: I will join you. [timestamp]: 3 February 2025 10:00:00<END_OF_CONV>
[Jack]: Great! I will pick you up at 8am.[Lucy]: Thank you. [timestamp]: 3 February 2025 10:00:00<END_OF_CONV>
[Jack]: Have you ever climbed before? [Lucy]: No, I have not. [timestamp]: 3 February 2025 10:00:00<END_OF_CONV>
[Jack]: I will teach you how to climb.[Lucy]: Thank you. [timestamp]: 3 February 2025 10:00:00<END_OF_CONV>


Output:
{
   "memories": [
         "Jack planned to go climbing the day after 3 February 2025, will pick Lucy up at 8am and teach her how to climb",
         "Lucy had never climbed before 3 February 2025 and agreed to join Jack for climbing"
   ]
}

# CRITICAL GUIDELINES:

## Time Processing Rules:
- Preserve relative time references but always include timestamp context:
  - Note that timestamp entry in input shows the time of the information being shared, not the time of event happening.
    You need to conver the time if happening time does not match the timestamp.
  - "tomorrow" + timestamp "3 February 2025" → "the day after 3 February 2025"
  - "next month" + timestamp "January 2025" → "the month after January 2025"
  - "two days ago" + timestamp "10 May 2025" → "two days before 10 May 2025"

## Memory Construction Rules:
- Each memory must be self-contained and independently understandable
- Include sufficient context (who, what, when, where, why)
- Combine related information from the same topic into single memories
- Separate different topics into different memory entries
- Use clear, natural language that preserves the original meaning

## Quality Control:
- Do not extract information from system messages
- Do not reference the examples provided in this prompt
- Return empty list if conversations cover unrelated topics with insufficient coherent information
- Ensure each memory entry provides actionable information for future reference

# OUTPUT FORMAT:
Always return a JSON object with a "memories" key containing an array of strings:
{
   "memories": ["memory1", "memory2", ...]
}
"""

ITERATIVE_ANWSER_PROMPT_VECMEM = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories. 
You will be given some memories and a question. Your job is to determine if the memories are sufficient to answer the question accurately. 
If the memories are not sufficient, you can determine to retrieve more memories from the memory store, 
by proving at most 1 extra query to the memory store. 
You should return a json object with two keys: 
"sufficient": "yes" or "no", 
"search_query": "<your query here>"
- If the memories are sufficient, you should return "sufficient": "yes" and "search_query": "". 
- If the memories are not sufficient, you should return "sufficient": "no" and "search_query": "<your query here>". 

Example: 

Query: Where will Bob go after gym? 

Memories: 
[Bob]: I went to the gym yesterday and then visited my friend Alice. [timestamp]: 2nd January 2024 
[Bob]: I feel happy about the gym exercise. [timestamp]: 2nd January 2024 

Output: { 
   "sufficient": "no", 
   "search_query": " Where does Alice live?", 
} 

In the example above, it is wrong to anwser "Bob visited Alice after gym". The correct anwser needs to clearly state the location of the visit. 
The new question is aksed because we know that Bob visited Alice after gym from memory, and we need to know the location of the visit.
You reconstructed question should also consider the content of the memories that have been retrieved using past questions.
Pay attention to this kind of connections and make reasonable retrieve.

Note that it might need to construct query in a very different way to infer the answer. 

You must obey the following rules: 
1. You must return a json object with the keys "sufficient","search_query"  
2. You should not use the examples provided in this prompt to answer the question. 
3. You can construct at most one search query. They must be concise, directly relevant, and not paraphrases or fragments of the question. 
4. You should never construct a query that is not related to the question or too general. 
5. Only construct new query when necessary. 
6. You should never use the question itself (or remove some constrains from the question) to construct new query. 
7. The system will also provide you with all the questions you asked before, you should NEVER construct the same queries or paraphrases of the ones you asked before. 

Question: 
{{question}} 
Memories: 
{{memories}} 
Questions you asked before: 
{{asked_questions}}
"""

ITERATIVE_ANWSER_PROMPT_VECMEM_REASON = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories. 
You will be given some memories and a question. Your job is to determine if the memories are sufficient to answer the question accurately. 
If the memories are not sufficient, you can determine to retrieve more memories from the memory store, 
by proving at most 1 extra query to the memory store. 
You should return a json object with three keys: 

"reason": "<your reason here>"
"sufficient": "yes" or "no", 
"search_query": "<your query here>"

- If the memories are sufficient, you should return "sufficient": "yes" and "search_query": "". 
- If the memories are not sufficient, you should return "sufficient": "no" and "search_query": "<your query here>". 
- For both cases, you should also return a reason for your decision, which might be helpful for other agents to understand the context better.

Example: 

Query: Where will Bob go after gym? 

Memories: 
[Bob]: I went to the gym yesterday and then visited my friend Alice. [timestamp]: 2nd January 2024 
[Bob]: I feel happy about the gym exercise. [timestamp]: 2nd January 2024 

Output: { 
   "reason": "The new question is aksed because we know that Bob visited Alice after gym from memory, but we need to know the location of the visit. Thus we only need to ask where does Alice live to get the anwser."
   "sufficient": "no", 
   "search_query": " Where does Alice live?", 
  
} 

You reconstructed question should also consider the content of the memories that have been retrieved using past questions.
Pay attention to this kind of connections and make reasonable retrieve.

Note that it might need to construct query in a very different way to infer the answer. 

You must obey the following rules: 
1. You must return a json object with the keys "reason","sufficient","search_query"  
2. You should not use the examples provided in this prompt to answer the question. 
3. You can construct at most one search query. They must be concise, directly relevant, and not paraphrases or fragments of the question. 
4. You should never construct a query that is not related to the question or too general. 
5. Only construct new query when necessary. 
6. You should never use the question itself (or remove some constrains from the question) to construct new query. 
7. The system will also provide you with all the questions you asked before, you should NEVER construct the same queries or paraphrases of the ones you asked before. 

Question: 
{{question}} 
Memories: 
{{memories}} 
Questions you asked before: 
{{asked_questions}}
"""

SEMANTIC_EXTRACTION_PROMPT = """

You are an expert system in semantic memory extraction. Each semantic memory is a fact that is expected to be persistent and valuable.


You will be given:
1. Raw reference: A concatenated message pieces from a conversation, which are expected to be related to the same topic.
2. Episodic Memory: Topic summary that is generated from the raw reference.
3. Old Semantic Memories: A list of semantic memories that might related the episodic memory, which are generated before.

Your job is to extract HIGH-VALUE, PERSISTENT, NEW semantic memories using Episodic Memory and Raw Reference.
The old semantic memories are provided to you for context reference, you can use them to capture precious connections.
If some information in the raw reference and the episodic memory have been captured in the old semantic memories, you should not generate redundant information

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, not temporary conversation details.

Instructions (Think step by step):
1. Extract ONLY knowledge that passes these tests:
   a. **Specificity Test**: Does it contain concrete, searchable information?
   b. **Utility Test**: Can this help predict future user needs?
   c. **Independence Test**: Can be understood without conversation context?
2. You should generate facts even if the information has been included in the episodic memory. 
3. You should not generate facts that are already included in the old semantic memories.
4. Pay EXTRA attention to the facts in the raw reference that are missing from both the episodic memory and the old semantic memories.
5. Be careful about time information. The raw reference contains some timestamps for each message piece, representing the time of the conversation happening.
   The time of when conversation happens does not necessarily represent the time of the event happening. What you need to pretain is the time of the event happening.


## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
   - Names, titles, companies, roles
   - Education, qualifications, skills
   
2. **Persistent Preferences**  
   - Favorite books, movies, music, tools
   - Technology preferences with reasons
   - Long-term likes and dislikes
   
3. **Technical Knowledge**
   - Technologies used (with versions)
   - Architectures, methodologies
   - Technical decisions and rationales
   
4. **Relationships**
   - Names of family, colleagues, friends
   - Team structure, reporting lines
   - Professional networks
   
5. **Goals & Plans**
   - Career objectives
   - Learning goals
   - Project plans
   
6. **Patterns & Habits**
   - Regular activities
   - Workflows, schedules
   - Recurring challenges

## Examples:

HIGH-VALUE (Extract these):
- "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
- "John has a dog named Rex"
- "Allen visited Roma on 1st January 2024"
- "Bob prefers PyTorch over TensorFlow for debugging"
- "Alice's team lead is named Sarah"
- "Bruce has been practicing yoga since March 2021"
- "James feel happy about the concert on 1st January 2024"

## Output Format

Return ONLY high-value knowledge in JSON format:
{
    "facts": [
        "First high-value persistent fact...",
        "Second high-value persistent fact...",
        "Third high-value persistent fact..."
    ]
}
If no new information is found, you should return an empty list in the "facts" key.

Quality over quantity - extract only knowledge that truly helps understand the user long-term.

Episodic Memory:
{{episodic_memory}}
Raw Reference:
{{raw_reference}}
Old Semantic Memories:
{{old_semantic_memories}}
Output:
"""

SEMANTIC_EXTRACTION_DURING_MERGE_PROMPT = """

You are an expert system in semantic memory extraction. Each semantic memory is a fact that is expected to be persistent and valuable.


You will be given:
1. Raw reference: A concatenated message pieces from a conversation, which are expected to be related to the same topic.
2. Episodic Memory: Topic summary that is generated from the raw reference.
3. Old Semantic Memories: A list of semantic memories that might related the episodic memory, which are generated before.

Your job is to extract HIGH-VALUE, PERSISTENT, NEW semantic memories in the Raw Reference.
The old semantic memories are provided to you for context reference, you can use them to capture precious connections.
If some information in the raw reference have been captured in the old semantic memories or the episodic memory, you should not generate redundant information

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, not temporary conversation details.

Instructions (Think step by step):
1. Extract ONLY knowledge that passes these tests:
   a. **Specificity Test**: Does it contain concrete, searchable information?
   b. **Utility Test**: Can this help predict future user needs?
   c. **Independence Test**: Can be understood without conversation context?
   d. **Redundancy Test**: Does it contain information that is already included in the old semantic memories or the episodic memory?
2. Pay EXTRA attention to the facts in the raw reference that are missing from both the episodic memory and the old semantic memories.
3. Be careful about time information. The raw reference contains some timestamps for each message piece, representing the time of the conversation happening.
   The time of when conversation happens does not necessarily represent the time of the event happening. What you need to pretain is the time of the event happening.


## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
   - Names, titles, companies, roles
   - Education, qualifications, skills
   
2. **Persistent Preferences**  
   - Favorite books, movies, music, tools
   - Technology preferences with reasons
   - Long-term likes and dislikes
   
3. **Technical Knowledge**
   - Technologies used (with versions)
   - Architectures, methodologies
   - Technical decisions and rationales
   
4. **Relationships**
   - Names of family, colleagues, friends
   - Team structure, reporting lines
   - Professional networks
   
5. **Goals & Plans**
   - Career objectives
   - Learning goals
   - Project plans
   
6. **Patterns & Habits**
   - Regular activities
   - Workflows, schedules
   - Recurring challenges

## Examples:

HIGH-VALUE (Extract these):
- "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
- "John has a dog named Rex"
- "Allen visited Roma on 1st January 2024"
- "Bob prefers PyTorch over TensorFlow for debugging"
- "Alice's team lead is named Sarah"
- "Bruce has been practicing yoga since March 2021"
- "James feel happy about the concert on 1st January 2024"

## Output Format

Return ONLY high-value knowledge in JSON format:
{
    "facts": [
        "First high-value persistent fact...",
        "Second high-value persistent fact...",
        "Third high-value persistent fact..."
    ]
}
If no new information is found, you should return an empty list in the "facts" key.

Quality over quantity - extract only knowledge that truly helps understand the user long-term.

Episodic Memory:
{{episodic_memory}}
Raw Reference:
{{raw_reference}}
Old Semantic Memories:
{{old_semantic_memories}}
Output:
"""


CONSOLIDATION_DECISION_PROMPT = """
You are a memory consolidation agent. Your task is to decide how to handle conversation memories.

## Your Capabilities

You have three actions:
1. **merge**: If the new memory is related to an existing episodic memory, merge them.
   - Use when: New memory topic matches existing episodic memory topic
   - Reference: MEMORY_AUGMENT_MERGE_PROMPT logic

2. **augment**: If multiple relevant vector memories exist, create a new episodic memory.
   - Use when: 3+ relevant raw memories found, no existing episodic match
   - Reference: MEMORY_AUGMENT_PROMPT logic

3. **none**: If no relevant memories found, store as raw vector memory.
   - Use when: No episodic match AND fewer than 3 relevant vector memories

## Input Format

You will receive:
- Current session dialogue (new memory to consolidate)
- Retrieved episodic memories (existing, for merge check)
- Retrieved vector memories (raw, for augment check)
- Retrieval scores

## Decision Rules

1. **Merge Check First**:
   - If retrieved episodic memory score >= 0.85 → MERGE
   - Otherwise continue to augment check

2. **Augment Check Second**:
   - If 3+ retrieved vector memories with score >= 0.7 → AUGMENT
   - Otherwise → NONE

## Important Rules

- Each memory must be self-contained with complete time information
- Only combine information from the same topic
- Preserve temporal accuracy (timestamp context)
- Return ONLY JSON, no additional text

## Output Format

Respond with JSON only (no extra text):
{
    "action": "merge" | "augment" | "none",
    "reasoning": "brief explanation of your decision",
    "target_episodic_id": <int, for merge only>,
    "relevant_vec_ids": <list of int, for augment only>,
    "merged_content": "<str, for merge: combined content>",
    "augmented_content": "<str, for augment: new episodic summary>"
}
"""


CONSOLIDATION_DECISION_WITH_QA_PROMPT = """
You are a memory consolidation agent. Your task is to:
1. Decide how to consolidate conversation memories
2. Answer questions based on the conversation context

## Your Capabilities

You have three consolidation actions:
1. **merge**: If the new memory is related to an existing episodic memory, merge them.
2. **augment**: If multiple relevant vector memories exist, create a new episodic memory.
3. **none**: If no relevant memories found, store as raw vector memory.

## Input

You will receive:
- Session dialogue (new memory to consolidate)
- Retrieved episodic memories (existing memories for reference)
- Retrieved vector memories (raw conversation pieces)
- Questions to answer (for evaluation)

## Consolidation Decision Rules

1. **Merge Check First**:
   - If episodic memory score >= 0.85 → MERGE
   - Otherwise continue to augment check

2. **Augment Check Second**:
   - If 3+ vector memories with score >= 0.7 → AUGMENT
   - Otherwise → NONE

## Questions

Answer each question based on the session dialogue and memories provided.
Your answers will be used to evaluate the quality of memory consolidation.

## Output Format

Respond with JSON only (no extra text):
{
    "action": "merge" | "augment" | "none",
    "reasoning": "brief explanation of your decision",
    "target_episodic_id": <int, for merge only>,
    "relevant_vec_ids": <list of int, for augment only>,
    "merged_content": "<str, for merge: combined content>",
    "augmented_content": "<str, for augment: new episodic summary>",
    "predicted_answers": [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"}
    ]
}

## Important Rules

- Return ONLY JSON, no additional text
- Include all question-answer pairs in predicted_answers
- For consolidation: follow topic coherence and temporal accuracy rules
- Each memory must be self-contained with complete time information
"""

# - Today's date is {datetime.now().strftime("%Y-%m-%d")}.