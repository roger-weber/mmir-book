# Chunking Strategies {#sec-rag-chunking}

Before a document can be retrieved for RAG, it has to be split into pieces. Embedding a whole book as a single vector is useless: the vector averages over so many topics that it is close to nothing in particular, and a query for one fact drowns in the rest. Splitting documents into smaller units, called chunks, is therefore the first design decision in any RAG system, and a surprisingly consequential one.

Chunking is older than RAG. Classical retrieval already split long documents so an engine could show the user the most relevant passage, with the matched terms close together, instead of a whole document to scroll through. RAG revived the technique for a harder reason: early embedding models lost accuracy beyond a few hundred tokens, and early language models had context windows too small to accept a full document. Chunking was the way to fit both.

Those technical limits have eased, but chunking has not become irrelevant. It has become a deliberate trade-off. Larger chunks give the language model more context to reason over, which helps it compose complete answers. Smaller chunks produce sharper embeddings, because a short, focused passage has one clear meaning rather than a blurred average. These pull in opposite directions, and the chunk size is where we balance them.

As [[sec-rag-pipeline]] noted, the strongest resolution is not to pick one size but to keep two levels at once: small chunks that embed cleanly and drive retrieval, and larger parent chunks that expand the context handed to the generator. Most of the methods below produce one level; hierarchical chunking, later in this section, is where the two levels come together, and it is the arrangement RAG systems most often use.

> [!note] The chunk-size sweet spot
> In practice, chunks of roughly 512 to 1024 tokens balance the two pressures: large enough to carry a coherent idea, small enough to embed cleanly. Treat this as a starting point, not a rule. Factual lookups favour shorter chunks, because the answer sits in a small span; summarization or cross-topic questions favour longer context, because the answer must connect ideas that short fragments would separate.

> [!warning] Chunk sizes here are in characters, not tokens
> The code below uses LangChain's `CharacterTextSplitter`, which counts characters, while the sweet spot above and a model's context window are measured in tokens. As the [[ch-rag|chapter introduction]] notes, a token is neither a character nor a word, and the ratio is language-specific. A rough English rule is four characters, or three-quarters of a word, per token, but set and measure chunk sizes in the tokens of the model we actually use, not in characters or words.

The sections below walk through chunking methods from the simplest to the most adaptive, using one running document throughout: the Project Gutenberg edition of *A Study in Scarlet*. Each method fixes a weakness of the one before it.

## Fixed-Size Splitting

The simplest method ignores the text's structure entirely. Split the document into words, then group words until each chunk reaches a target size. To avoid cutting an idea cleanly in half at a boundary, let consecutive chunks overlap by a fixed amount, so a sentence split across the boundary still appears whole in one of the two chunks.

LangChain, one of the most widely used RAG frameworks, provides this directly:

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator=" ",          # break on word boundaries
    chunk_size=1000,        # target characters per chunk
    chunk_overlap=200,      # characters shared between neighbours
    add_start_index=True,   # record each chunk's offset in the source
)
chunks = text_splitter.split_text(study_in_scarlet)
```

Applied to the running document, this produces 299 chunks that pack tightly against the limit: every one lands between 889 and 1000 characters, almost all within a few characters of 1000, as [[fig-chunk-fixed]] shows. The small variation comes only from the splitter stopping at the last whole word before the limit. The method is easy and fast, and it is enough when rough locality is all we need.

![[fig-chunk-fixed]]

> [!warning] Fixed-size chunks ignore meaning
> Chunk boundaries fall wherever the character count runs out, not where topics change. A new section can start halfway through a chunk and finish in the next, so a single chunk can mix the end of one idea with the start of an unrelated one. Embeddings of such chunks blend concepts, and retrieval returns passages that lack coherent context.

## Sentence Boundaries

A first improvement is to treat the sentence, not the character count, as the smallest unit. We add whole sentences to a chunk until it reaches the size limit, so chunks never break mid-sentence. Chunk sizes vary a little as a result, but the pieces read more naturally and map better onto meaning.

Detecting sentence boundaries is harder than splitting on every period, because abbreviations like "Dr." and quoted dialogue create false boundaries. The mechanics of sentence segmentation, including the Punkt and spaCy segmenters, are covered in [[sec-advanced-text-processing-tokenization|Advanced Text Processing]]. Here we simply use such a segmenter as a building block:

```python
from langchain_text_splitters import NLTKTextSplitter  # or SpacyTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(study_in_scarlet)
```

On the running document, sentence-based splitting also yields 299 chunks, clustered just below the 1000-character limit with a mean near 930, as [[fig-chunk-sentence]] shows. The spread is wider than for fixed-size splitting, reaching down to about 560 characters, because a chunk closes as soon as the next whole sentence would overflow it, and sentence lengths vary. Sentence-based chunking balances structure against simplicity, but it still does not know where topics shift, so a single chunk can hold two unrelated but adjacent sentences.

![[fig-chunk-sentence]]

## Splitting on Structure

Authors already mark where content changes: chapters, sections, paragraphs. Splitting on those structural elements produces chunks that align with the author's own units of thought. The signals depend on the format. In the Project Gutenberg text, more consecutive newlines mean a larger break, so four newlines mark a chapter and two mark a paragraph. In HTML or Markdown, heading tags and paragraph tags play the same role.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n\n\n", "\n\n"],  # four newlines = chapter, two = paragraph
    chunk_size=2000,     # large cap: keep each structural unit whole,
    chunk_overlap=0,     # only splitting a unit that exceeds the cap
    add_start_index=True,
)
chunks = text_splitter.split_text(study_in_scarlet)
```

Structural chunks are more coherent than word- or sentence-based splits, but their sizes are far less uniform, as the three panels of [[fig-chunk-structure]] show. Splitting by chapter yields just 19 pieces (top panel), most of them enormous, between 10,000 and 23,000 characters, with a handful of tiny ones from front matter and section breaks. Such coarse chunks are too big to embed well or to sit alongside others in the model's context. Splitting by paragraph controls size better and produces 816 pieces (middle panel), but *A Study in Scarlet* is written as dialogue, so more than half the paragraphs are under 200 characters, often a single line of speech. Those tiny chunks carry too little context to retrieve or answer from.

The fix is to merge. We keep paragraph splitting but merge adjacent small chunks until each one is substantial. On the running document this lifts the typical chunk to around 850 characters and clears out the dialogue fragments (bottom panel of [[fig-chunk-structure]]), though a few long paragraphs still run past 1500. For overlap, a fixed character count works poorly here, because a whole paragraph can be longer than the overlap window. A better approach adds the last sentence of the previous chunk and the first sentence of the next to each chunk, carrying just enough neighbouring context across the boundary.

![[fig-chunk-structure]]

## Semantic Splitting

Structural splitting assumes reliable structure, and that assumption breaks on scraped web pages, badly laid-out PDFs, and scanned documents with no headings. Semantic splitting drops the reliance on formatting and groups sentences by meaning instead.

The method extends sentence-based splitting with a similarity test. We embed each sentence, then merge neighbouring sentences or chunks when they are similar enough and the result stays under the maximum size. Similarity must be measured in meaning, not surface words: a term-frequency vector would score two differently-worded sentences about the same idea as unrelated, so we use a bi-encoder (see [[sec-semantic-similarity-with-transformers]]) and compare sentence embeddings with cosine similarity.

Merging proceeds iteratively. Start with each sentence as its own chunk; compute similarities between adjacent chunks; merge the most similar neighbouring pairs that stay within the size limit; repeat. Over several passes, this grows longer chunks wherever the text stays on one topic and leaves a boundary wherever the topic shifts. Semantic chunks align with meaning better than any fixed rule, at the cost of the extra computation to embed every sentence and compare neighbours.

The top panel of [[fig-chunk-semantic]] shows the result on the running document, and its shape is the signature of semantic splitting. The 491 chunks spread across the whole size range instead of piling up at the limit, because a boundary falls wherever the topic shifts, not where a character budget runs out, so the sizes track the content and the median lands near 540 characters. The cluster of very small chunks on the left is the method's cost: a sentence that resembles neither neighbour is left unmerged, so roughly a third of the chunks stay under 200 characters.

A simple refinement removes that cost. After the similarity merge, we make one more pass over the chunks that are still too small and fold each one into a neighbour, choosing the semantically closer of the two, and coalescing runs of adjacent fragments (the back-and-forth of dialogue) with each other first. This pass relaxes the similarity threshold, since the goal is now to eliminate fragments rather than to find strong matches, but it keeps a size ceiling. On the running document it cuts the 491 chunks to 355, removes every chunk under 200 characters, and lifts the median to about 670, as the bottom panel shows. A few chunks run slightly past 1000, where a fragment was attached to an already-full neighbour.

![[fig-chunk-semantic]]

## Hierarchical Chunking

The methods above each pick one chunk size, and that single choice cannot serve both retrieval and generation well. Hierarchical chunking refuses the choice and stores the document at more than one level of granularity at once. This is the arrangement most suited to RAG.

The idea is to split coarsely, then finely, and keep both. Divide the document by large structural units such as headings or sections. Within each unit, apply a finer splitter: sentence-based, fixed-token, or semantic. Store both levels, and give each a job. Small chunks produce accurate embeddings and drive retrieval; the larger section they belong to is passed to the language model as context, so generation sees the full surrounding passage even though retrieval matched on a precise fragment.

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
)
sections = header_splitter.split_text(long_markdown_text)

token_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
final_chunks = []
for sec in sections:
    for idx, sub in enumerate(token_splitter.split_text(sec.page_content)):
        final_chunks.append({
            "content": sub,
            "metadata": {**sec.metadata, "sub_chunk": idx},
        })
```

Retrieval runs against the small `sub` chunks; the stored metadata lets the system recover the parent section and hand that larger unit to the generator.

## Query-Aware Chunking

Every method so far splits the document before any query arrives, which is why it is called static chunking. Yet different questions need different amounts of context. "When was Albert Einstein born?" wants one short, fact-bearing chunk; "Summarize how vector search works" wants a whole section. Query-aware, or adaptive, chunking decides how much context to assemble only once it has seen the query.

The pattern builds on hierarchical storage. Preprocess the document into small, embedding-friendly chunks. When a query arrives, interpret its intent, retrieve the top chunks as usual, and then assemble context to match the intent: for a factual question, pass only the one or two highest-scoring chunks; for a summary, combine neighbouring chunks or return the whole parent section; for a multi-hop question, gather related chunks from several sections and stitch them together.

This reduces noise for simple questions and supplies enough context for complex ones, while sending the language model fewer tokens when fewer will do. Frameworks like LangChain, LlamaIndex, and Haystack already implement early versions, retrieving small chunks to score relevance and promoting larger parent sections when needed. As context windows keep growing, query-aware chunking shifts from a workaround for size limits toward a way to optimize relevance, cost, and reasoning quality.

## Which Strategy Is Best?

After six methods, the natural question is which one to use, and the honest answer is that there is no universal winner. The best strategy depends on the documents, the questions users ask, and how much context the reader can absorb. What we can do is measure, but the metrics from [[ch-evaluation]] do not transfer as cleanly as they first appear.

The difficulty is that relevance is defined for a query, while chunking happens before any query exists. The chunk boundaries, fixed in advance, then decide how well retrieval can ever do. If an answer straddles a boundary, no single chunk holds it whole, and recall suffers no matter how good the retriever is. If a chunk buries one relevant sentence in paragraphs of unrelated text, the document looks relevant but the context handed to the generator is dilute. Chunking quality is really a question of how well boundaries line up with answers, and a histogram of chunk sizes, useful as the plots above are for spotting tiny or oversized chunks, cannot tell us that.

The dependable way to compare strategies is therefore to measure their effect on the task, holding the rest of the pipeline fixed and varying only the chunking. Against a set of questions with known answers, the key metric is answer recall at $k$: the share of questions for which at least one of the top $k$ retrieved chunks contains the answer. It counts a hit per question, unlike the document recall of [[sec-performance-evaluation-precision-recall|Performance Evaluation]], which counts how many of all relevant documents were found, and it is the form of recall that bad boundaries break. A second axis is context precision, or token efficiency: how much irrelevant text rides along in those chunks, since large chunks win recall but waste the context budget and can distract the reader. The final judge is the end-to-end answer quality, measured with the faithfulness and context metrics of [[sec-rag-evaluation]].

A cheaper, label-free diagnostic is sometimes enough during development: compare the similarity of sentences within a chunk against the similarity across a boundary. A good split keeps sentences inside a chunk alike and drops sharply at the boundary, which is exactly the cohesion that semantic splitting optimizes. It is a useful smell test, but no substitute for measuring the task itself. In the end the right chunking is the one that scores best on our own questions and our own reader, not the one with the tidiest distribution.

> [!demo] Chunking strategies {notebook="https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-01-chunking-strategies.ipynb"}
> Split one long document with fixed-size, sentence, structural, semantic, hierarchical, and query-aware chunking, compare the chunk-size distributions, and rank the strategies by answer recall at $k$ against the context each one costs.
>
> *Includes pre-run results: you can read through or download and experiment.*
