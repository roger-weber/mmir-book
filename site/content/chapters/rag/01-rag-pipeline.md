# The RAG Pipeline {#sec-rag-pipeline}

The previous section made the case for answering the question instead of returning a list, by pairing a retriever with a generator. This section lays out the pipeline that does it. Most of that pipeline is machinery we have already built: the inverted index and BM25 from [[ch-index]], the embeddings from [[sec-word-embeddings]], and the vector search and cross-encoder reranking from [[ch-vector]] and [[sec-semantic-similarity-with-transformers]]. Two parts are genuinely new to RAG, and each gets its own section after this one: how documents are split into chunks at two levels, and how a final stage turns retrieved passages into an answer. We start with the shape of the whole pipeline, then isolate those two additions.

A direct answer is not a new idea. Long before large language models, web search engines were already moving from lists of links toward answers a user could read without clicking through. Understanding that lineage makes the modern RAG pipeline easier to read, because each stage of it automates a step that search engines once did by hand.

> [!note]- From Knowledge Graph to AI Overview: how search learned to answer (optional reading)
> Google's Knowledge Graph, launched in 2012, was an early form of retrieval augmentation. Rather than returning raw documents for a query like "Albert Einstein", it supplied structured, authoritative facts: birth date, field, key contributions, related people. The answer came from a curated knowledge base, not from the ranked documents, and it shifted search from "find the page" toward "state the fact".
>
> Featured Snippets followed in 2014. Here the system retrieved candidate pages the usual way, then scanned them for the passage that best answered the query and promoted that passage above the organic results. This is extractive question answering deployed at web scale: query understanding, document retrieval, passage extraction, answer selection. The answer was copied verbatim from a source, so it was grounded by construction, but the system could not rephrase or combine passages.
>
> AI Overview, introduced in 2024, added the generative step. It retrieves relevant web content and passes it to a language model, which composes a fluent answer grounded in those sources. The three features trace the same arc this chapter follows: from structured facts, to extracted passages, to generated-but-grounded answers.
>
> ![[fig-rag-figure-7-1]]

## Why Retrieve at All

A language model already contains an enormous amount of knowledge in its weights. The reason to retrieve is that this parametric knowledge has three hard limits, and all three are fixed by supplying external text at query time rather than by training.

The first limit is time. Weights are frozen at training, so anything that happened afterwards, like this year's Nobel laureates, is simply absent. The second is scope. No general model has read your company's internal policies, a specific insurance product's terms, or a private document collection, and these are exactly the sources that specialized questions depend on. The third is trust. A generation-only model produces fluent text whether or not it has the facts, so it can state a confident falsehood, a failure we call a hallucination. [[fig-rag-vs-generation-only-comparison]] shows the time limit in action: asked about an event past its training cutoff, a generation-only model cannot answer, while a retrieval-grounded system answers from current sources.

![[fig-rag-vs-generation-only-comparison]]

Fine-tuning the model on the missing material is one response, but it is costly, slow, and stale the moment the underlying facts change. RAG is the cheaper and more flexible option: leave the model fixed and change what we put in front of it. Because the answer is now conditioned on retrieved passages, the model is pulled toward evidence and away from invention, and the system can cite the passages it used so a reader can check them.

![[fig-rag-architecture]]

The minimal shape of RAG is small, shown in [[fig-rag-architecture]]. A query goes two ways at once. It prompts the generator directly, and it queries a document store, which returns the most relevant snippets. The generator then composes its answer from both the question and those snippets. Everything else in this chapter is a refinement of this loop: better ways to prepare the documents, to filter and rank what comes back, and to turn it into a trustworthy answer.

## From Two Steps to a Pipeline

Early RAG was literally two steps: retrieve, then generate. It worked, but it inherited the weaknesses of each retrieval method on its own. A keyword search with BM25 (see [[sec-indexing-ranked-retrieval|Ranked Retrieval]]) matches exact terms, so a document that says "the 2025 Nobel Physics laureate was awarded to" can be missed by the query "who won the Nobel Prize", because "won" and "was awarded" share no tokens. Dense vector search (see [[ch-vector]]) matches meaning and bridges that gap, but it blurs fine detail: the word "who", which signals that the user wants a person, dissolves into the sentence embedding, so the system may return pages about the prize rather than the people who won it.

Modern systems answer this by chaining several stages, each repairing the previous one's weakness. [[fig-advanced-rag-pipeline-architecture]] shows the full pipeline for an informational query. We walk through it with one running example:

> Who won the Nobel Prize in Physics in 2025?

![[fig-advanced-rag-pipeline-architecture]]

**Query expansion** reshapes the raw query so retrieval can succeed. It may fix spelling, add synonyms, rewrite the question, or break it into sub-questions. Our example might expand to variants like "2025 Physics Nobel winners" and "physics Nobel laureates 2025". The classical techniques for this were covered in [[sec-advanced-text-processing-query-understanding|Advanced Text Processing]]. Language models add rewriting patterns of their own: HyDE writes a plausible answer and searches for that instead of the question, multi-query generates several rewrites and merges their results, and step-back prompting first generalizes the query by one level. In a modern system the model itself performs the rewrite, which we return to in [[sec-rag-agentic]].

**Sparse and dense retrieval** run the expanded query through the two complementary engines from earlier chapters. Sparse retrieval (BM25) returns matches by exact terms; dense retrieval returns matches by embedding similarity, even with different wording. Neither should return whole documents, because the generator needs a focused context of what actually matters rather than twenty pages of a PDF to wade through. Where the two engines differ is in what they demand of the split. Dense retrieval needs genuinely small chunks, since the embedding of a long passage blurs its meaning; sparse retrieval has no such need and matches terms in text of any length. RAG reconciles this with two levels: small chunks that embed cleanly and drive dense matching, and a larger parent chunk, still focused rather than a whole document, that is handed to the generator so it has enough to answer from. This two-level split is the subject of [[sec-rag-chunking]].

**Hybrid retrieval** merges the two result sets into one candidate pool, typically with a rank-fusion method such as reciprocal rank fusion (RRF), or by running cheap sparse retrieval first and refining its top results with the dense model. Merging is less trivial than it looks once the two engines work at different chunk levels. Rank fusion assumes both lists rank the same kind of item, so the pipeline must first bring them to a common unit, usually the parent chunk: the dense engine resolves each small chunk it matched up to the parent it belongs to, while sparse retrieval already works at that larger level. Because several small chunks often resolve to the same parent, those duplicates have to be collapsed into a single entry, keeping its best rank or combining the scores, before fusion runs. Only then do the sparse and dense rankings line up on the same chunks and RRF can combine them. Either way the pipeline keeps a few hundred candidates where a single retriever would have kept a biased few.

**Reranking** applies a cross-encoder (see [[sec-semantic-similarity-with-transformers]]) to the merged candidates. Unlike the bi-encoders used for retrieval, which embed query and document separately, a cross-encoder reads the query and a candidate together and scores their relevance directly. It is accurate enough to recognize that "who" calls for a person, and expensive enough that we apply it only to the shortlist, cutting a few hundred candidates down to a handful.

**The result-generation stage** turns those top passages into an answer, and it comes in three forms. A reader marks the exact span in a passage that answers the question. A generator composes a fluent, direct answer from the passages. A synthesizer summarizes across several documents, gathering evidence over multiple rounds. These three are the subject of [[sec-rag-result-presentation]].

**Guardrails** sit at the edges of the pipeline rather than at one point in it. They constrain what enters (rejecting unsafe or out-of-scope queries) and what leaves (checking that the answer is supported by retrieved evidence before it reaches the user). The **multi-hop** loop lets the stage send a follow-up query back to the start when one retrieval pass is not enough. Both become central once an agent, rather than a fixed pipeline, drives the loop, as we see in [[sec-rag-agentic]].

## What RAG Adds

Reading the pipeline back, most of it is familiar. Query expansion builds on the query understanding of [[ch-advanced-text]]; sparse, dense, hybrid retrieval and reranking are exactly the components of Chapters 4 to 6. Two stages are new, and they are new because the consumer of retrieval has changed from a human skimming a list to a model composing an answer.

The first is two-level chunking. A human reader is content with a snippet, but a generator needs enough surrounding text to compose a correct answer. RAG resolves this by splitting documents twice: small chunks that embed cleanly and drive precise retrieval, and larger parent chunks that are handed to the model to give it room to reason. The second is the result-generation stage itself: the reader, generator, and synthesizer that convert retrieved passages into a marked span, a direct answer, or a cross-document summary. Neither stage existed when the output was a ranked list. The next two sections take them in turn.

## A Modular, Tunable System

The value of drawing RAG as a pipeline is that every stage is optional and swappable. A simple factual lookup can skip dense retrieval and reranking entirely and still answer correctly. A hard, domain-specific question can afford the full chain. The same architecture therefore spans a fast, cheap search tool and a slow, thorough research assistant, by turning stages on or off rather than by building a different system.

Each stage also makes a speed-for-quality trade. Sparse retrieval is fast but shallow on meaning; dense retrieval captures meaning but costs more compute; reranking raises precision but is the most expensive step per candidate. Running all three gives the best answers at the highest cost, which is why production systems route easy queries down a short path and reserve the full pipeline for queries that need it.

One consequence is worth stating plainly, because it reflects a real shift in what users will accept. Classical search chased sub-ten-millisecond latency at almost any cost. RAG relaxes that. A streamed answer that arrives over a few seconds feels natural, and for a genuinely better answer, especially in research settings, users tolerate far longer waits than a link-ranking engine could ever get away with. That tolerance is what makes a multi-stage pipeline affordable in the first place.

## Building the Pipeline in a Framework {#sec-rag-frameworks}

Modularity is also what makes the pipeline practical to build. Wiring retrievers, rerankers, and generators together by hand is slow and error-prone, so frameworks such as Haystack, LangChain, and LlamaIndex assemble them from reusable components. We use Haystack because its design mirrors [[fig-advanced-rag-pipeline-architecture]] directly: every stage is a component with typed inputs and outputs, and a pipeline is the graph that connects them.

A Haystack system has two pipelines. An indexing pipeline fills a document store: a splitter cuts documents into chunks, an embedder computes their vectors, and a writer stores both. The store can be an in-memory one for experiments or a production backend such as OpenSearch or a vector database, the systems of [[sec-indexing-scaling-out|Index for Text Retrieval]] and [[sec-vector-search-in-practice|Vector Search]]. The query pipeline then answers questions against that store. The sketch below builds the hybrid core of the pipeline above: BM25 and dense retrieval in parallel, fused with reciprocal rank fusion, reranked by a cross-encoder, and passed to a generator.

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.sentence_transformers import (
    SentenceTransformersSimilarityRanker,
)
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

store = InMemoryDocumentStore()   # filled beforehand by the indexing pipeline

template = [ChatMessage.from_user("""Answer the question using only the passages below.
{% for doc in documents %}[{{ loop.index }}] {{ doc.content }}
{% endfor %}
Question: {{ query }}""")]

rag = Pipeline()
rag.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=50))
rag.add_component("embedder", SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"))
rag.add_component("dense", InMemoryEmbeddingRetriever(document_store=store, top_k=50))
rag.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
rag.add_component("rerank", SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5))
rag.add_component("prompt", ChatPromptBuilder(template=template,
                                              required_variables=["query", "documents"]))
rag.add_component("llm", AmazonBedrockChatGenerator(model="us.anthropic.claude-sonnet-5-5"))

rag.connect("embedder.embedding", "dense.query_embedding")
rag.connect("bm25", "join")
rag.connect("dense", "join")
rag.connect("join", "rerank")
rag.connect("rerank", "prompt.documents")
rag.connect("prompt", "llm")

question = "Who won the 2025 Nobel Prize in Physics?"
result = rag.run({
    "bm25": {"query": question},
    "embedder": {"text": question},
    "rerank": {"query": question},
    "prompt": {"query": question},
})
print(result["llm"]["replies"][0].text)
```

The Sentence-Transformers components and the model generators ship in separate integration packages (`haystack_integrations`), and the generator here is a Claude model on Amazon Bedrock. Each component maps onto a stage of the pipeline, and each can be swapped without touching the rest: a different embedding model, a stronger reranker, a production store, or another generator is a one-line change. Here both retrievers search the same chunks, so the joiner can fuse their rankings directly. With the two-level chunking of [[sec-rag-chunking]], Haystack offers a hierarchical splitter and an auto-merging retriever that match small chunks and return their parent when enough of its children are relevant, which is the reconciliation to a common unit discussed above.

The same structure is also the pipeline's limitation. The graph is fixed when it is built, and it encodes one way of answering one kind of question. A different intent needs a different graph, and a change means new code. [[sec-rag-agentic]] returns to this problem.

> [!demo] The RAG pipeline {notebook="https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-02-rag-pipeline.ipynb"}
> Build the pipeline of this section stage by stage on a Wikipedia knowledge base about the 2025 Nobel Prizes: two-level indexing, BM25 and dense retrieval, reciprocal rank fusion, cross-encoder reranking, and a grounded answer, then wire the same pipeline in Haystack.
>
> *Includes pre-run results: you can read through or download and experiment.*
