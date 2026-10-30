# Retrieval Augmented Generation

Every retrieval system so far has produced the same kind of output: a ranked list. Chapters 1 through 6 made that list better, more relevant, faster to compute, ranked by meaning as well as by words, but what the user receives is still a set of documents to open and read. The system finds where the answer probably is, and the user does the last step of extracting it.

For many searches that is exactly right. Someone shopping for a high-end gaming PC wants options to compare, not a single verdict, and a ranked list of products is the natural answer. Other questions, though, have one answer, and a list buries it. Ask "who won the 2025 Nobel Prize in Physics" and ten links with the names buried three paragraphs down is a poor experience. Here the user wants the answer, not the haystack that contains it, and the better experience is to read the results for them and return the answer directly.

Chat agents reach the same goal from the opposite direction. A language model already answers questions and synthesizes fluent replies, but only for what it saw during training. It cannot answer about this year's Nobel Prize, a company's current policies, or a document it has never read. To do that it must reach outside itself: call a search tool over a knowledge base, retrieve the relevant material, and extract or synthesize the answer from it. Grounding generation in retrieved documents this way is what the term Retrieval-Augmented Generation describes. Early RAG simply built a context by placing retrieved results in front of the model. Modern RAG is agentic: the model issues tool calls inside a reasoning loop and decides for itself when and what to retrieve. One consequence matters for this book. The model can now rewrite the query before calling the search tool, replacing the hand-built, hard-coded query transformations of earlier chapters with a learned one.

![[fig-rag-two-paradigms]]

Put the two directions together, as in [[fig-rag-two-paradigms]], and each fixes the other's weakness. Retrieval spares the agent from reading a whole collection, by handing it only the snippets needed to form the answer. The agent spares the user from reading the result list, by turning those snippets into a direct reply. The retrieval system supplies grounded evidence, and the generator supplies the fluent, synthesized answer.

That leaves one new problem, and it is the hard one. A language model is trained to produce an answer, and it will produce a confident one whether or not it has the facts, inventing plausible but false detail. We call this failure a hallucination. Much of what makes RAG more than "search plus a chatbot" is the machinery that keeps the generated answer tied to the retrieved evidence rather than to the model's urge to answer.

This chapter extends retrieval at the user-experience front: instead of returning only a ranked list, we want to answer the question, or surface the answer, directly. Google already does both. For a question-like query it can highlight the sentence in a top result that answers it, with no language model involved, just a direct answer lifted from the page. Its newer AI mode goes further and uses a language model to answer and elaborate. We look at how this goal reshapes retrieval, in chunking and in result generation, and how to build the whole use case end-to-end as a pipeline.

> [!note] A note on units: characters, tokens, and words
> This chapter mixes three ways of measuring text, and they do not line up. Humans count words. Chunking, as we do it in the next sections, often counts characters. Embedding models and the language model itself count tokens, the subword units a tokenizer produces (see Sub-Word Tokenization in [[sec-transformer-embeddings]]), for both the context they read and the text they generate. A token is neither a character nor a word, and the conversion is not even constant across languages. A tokenizer is trained mostly on English, so it is thriftier there: the GPT-3.5/4 tokenizer encodes the English sentence "Information retrieval finds the documents that answer a question." as 10 tokens for 9 words, about 1.1 tokens per word, while its Italian translation "Il recupero delle informazioni trova i documenti che rispondono a una domanda." takes 21 tokens for 12 words, about 1.75 tokens per word. The same meaning costs more than twice the tokens, so a fixed token budget, whether a chunk size or a context window, holds much less text in some languages than in others.

> [!tip] What you'll learn
> - Explain why a ranked list falls short when the user wants a direct answer, and how RAG couples retrieval with a generative agent to answer and ground questions the model has not seen.
> - Trace the end-to-end RAG pipeline and identify how it reuses indexing, embeddings, and reranking from earlier chapters.
> - Compare chunking strategies and choose chunk sizes that serve both retrieval and generation.
> - Distinguish the reader, generator, and synthesizer that turn retrieved passages into a marked span, a direct answer, or a cross-document summary.
> - Explain how an agentic RAG loop takes over intent routing and query transformation, and how input and output guardrails keep it safe and grounded.
> - Evaluate RAG quality with faithfulness and context metrics, and judge when RAG is not the right tool.

> [!seealso] Prerequisites
> - [[ch-index]]: the inverted index and BM25, which supply the sparse-retrieval arm of a RAG pipeline.
> - Semantic Search: embeddings for meaning and cross-encoder reranking. See [[sec-word-embeddings]] and [[sec-semantic-similarity-with-transformers]].
> - [[ch-vector]]: approximate nearest-neighbour search over embeddings, the dense-retrieval arm.
> - [[sec-advanced-text-processing-tokenization|Advanced Text Processing]]: tokenization and sentence segmentation, reused when splitting documents into chunks.
> - [[ch-evaluation]]: precision, recall, and nDCG, which reappear when we measure retrieval quality inside RAG.
