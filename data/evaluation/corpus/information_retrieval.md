\# Information Retrieval



\## Retrieval



Information retrieval is the process of finding relevant information in a collection of documents given a user query.



A retrieval system typically converts documents and queries into representations and then ranks candidate documents according to estimated relevance.



\## TF-IDF



TF-IDF combines term frequency and inverse document frequency.



Term frequency measures how often a term occurs in a document. Inverse document frequency reduces the importance of terms that occur in many documents.



The resulting representation gives more weight to terms that are important to particular documents.



\## BM25



BM25 is a probabilistic lexical ranking function. It considers query-term frequency, document length, and the rarity of terms across the collection.



BM25 is especially useful when exact words or technical identifiers matter.



\## Dense Retrieval



Dense retrieval represents text as vectors in a continuous embedding space. A query embedding can be compared with document embeddings using a similarity function such as cosine similarity.



Dense retrieval can identify semantically related text even when the query and document use different wording.



\## Hybrid Retrieval



Lexical and dense retrieval provide complementary signals. A hybrid system can combine their ranked outputs.



Reciprocal Rank Fusion combines rankings by assigning a contribution based on the position of an item in each result list. This avoids directly comparing scores from different retrieval algorithms.



\## Reranking



A reranker receives an initial candidate set and evaluates the relationship between the query and each candidate more deeply.



A common architecture retrieves dozens of candidates efficiently and then uses a cross-encoder to rerank a smaller subset.



\## Evaluation



Recall@K measures how many relevant items appear in the top K results relative to all relevant items.



Mean Reciprocal Rank focuses on the position of the first relevant result.



nDCG evaluates ranking quality using discounted gains, giving greater importance to relevant items appearing near the top of the ranking.



\## Retrieval-augmented Generation



Retrieval-augmented generation combines a retrieval system with a language model. The retriever selects evidence and the generator uses that evidence to construct a response.



Grounding the answer in retrieved evidence can reduce unsupported responses and allows the application to expose sources alongside generated answers.

