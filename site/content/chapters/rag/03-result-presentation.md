# Result Generation {#sec-rag-result-presentation}

By the time the pipeline reaches this stage, retrieval and reranking have produced a short list of relevant passages. We could stop and show that list, but the point of RAG is to go further and return an answer. The stage that turns passages into an answer comes in three forms, and the difference between them is what the user gets back. We carry one running query through all three:

> Who won the Nobel Prize in Physics in 2025?

A **reader** marks the answer inside a passage and returns that span verbatim. A **generator** writes a new, direct answer in natural language from the retrieved passages. A **synthesizer** summarizes across several documents, gathering evidence over multiple rounds to build an answer no single passage contains. The three trade faithfulness for flexibility: the reader cannot say anything its source did not, which makes it safe but rigid; the synthesizer can assemble an answer from many sources, which makes it powerful but the hardest to keep honest.

## Choosing Among the Three

The right form depends on the question and the stakes, not on which is most sophisticated. Four factors decide it.

The size and quality of the context comes first. A reader works best on small, highly relevant passages where the answer is present verbatim; a generator or synthesizer suits broad, diverse context that must be combined. Question complexity follows: a simple fact wants a reader, while an open-ended or multi-part question wants a generator or synthesizer. User intent matters too: a user who wants an exact quote or citation is served by a reader, one who wants an explanation by a generator. Finally, risk tolerance can override the rest. In legal or medical settings, where an invented detail is dangerous, a reader's inability to add anything is a feature, not a limitation.

In a production system the choice is made per query. A classifier or the agent itself reads the query, picks the form, and plans the rest of the pipeline accordingly, which is the routing we return to in [[sec-rag-agentic]].

## The Reader

A reader finds the span of text in a passage that answers the question and marks it. It generates nothing; it predicts where the answer starts and where it ends. This is the mechanism behind a highlighted search result, and the standard design, built on BERT, is shown in [[fig-bert-extractive-qa-span-prediction]].

![[fig-bert-extractive-qa-span-prediction]]

The question and passage are joined into one sequence, marked by special tokens: `[CLS]` begins the input and `[SEP]` separates question from passage. BERT encodes the whole sequence, so each token's representation reflects both the question and the rest of the passage. A lightweight prediction head then assigns every passage token two scores, one for how likely it is to begin the answer and one for how likely it is to end it. The answer is the span whose start and end scores sum highest. The model is fine-tuned on datasets such as SQuAD, which provide the correct start and end positions for each question-passage pair, teaching the head to locate answer spans.

In code, we encode the question and passage together, run the model, and read off the span between the highest-scoring start and end tokens:

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForQuestionAnswering.from_pretrained(name)

question = "Who won the Nobel Prize in Physics in 2025?"
context = (
    "John Clarke, Michel Devoret, and John Martinis won the 2025 Nobel "
    "Prize in Physics for their discovery of macroscopic quantum tunnelling."
)

enc = tokenizer(question, context, return_offsets_mapping=True, return_tensors="pt")
offsets = enc.pop("offset_mapping")[0]
with torch.no_grad():
    output = model(**enc)

start = output.start_logits.argmax()   # highest-scoring start token
end = output.end_logits.argmax()       # highest-scoring end token
answer = context[offsets[start][0]: offsets[end][1]]
print(answer)
# John Clarke, Michel Devoret, and John Martinis
```

The model returns the exact span from the passage, with the start and end positions peaking sharply. Its strength is also its ceiling: it can only return text that is literally present, so it cannot combine two half-answers or rephrase an awkward one. This is why the Featured Snippet of [[sec-rag-pipeline]] could surface a direct answer without any language model at all: a reader only has to point, not to write.

## The Generator

A generator uses a language model to compose an answer token by token from the question and the retrieved passages. Because it writes rather than copies, it can merge facts from several passages into one fluent, direct response. Asked "what contributions led to the 2025 Nobel Prize in Physics?", it can draw on several documents and produce a coherent paragraph that no single passage states outright.

That freedom is also where hallucination enters, so the prompt is where a generator is controlled. Effective prompts keep the model grounded in the retrieved text and inside the intended domain. A well-built prompt states the task clearly (answer, summarize, or explain); supplies the retrieved passages as the only source of truth; reminds the model to answer only from that context and not from prior knowledge; restricts the scope to the application's domain; specifies the output format, such as a short answer, a paragraph, or JSON; optionally sets tone or audience; and tells the model what to do when the answer is not in the context, so it can say it does not know instead of inventing one.

How these pieces are arranged matters as much as whether they are present. During training a model learns particular ways of attending to the instruction, the question, and the context, and it is sensitive to their order and phrasing: burying the instruction after a long context, or leaving a key constraint implicit, can quietly degrade the answer even when all the information is there. Getting this right is the craft known as prompt engineering, and it is enough of a craft that we do not attempt to teach it here. The point to carry away is that the prompt is not boilerplate but a real lever on quality, worth iterating on. One practical safeguard is meta-prompting: using a language model to draft and refine the prompt itself against examples, which keeps prompts effective as models and requirements change rather than relying on a hand-tuned string that silently goes stale.

To make the building blocks concrete, the skeleton below lays them out in the structure Claude follows most reliably.

> [!note]- A grounded-answer prompt skeleton (optional reading)
> The retrieved passages go in tagged blocks and before the question, the grounding and citation rules are stated plainly, and the key constraint is repeated once after the data. The parts in braces are filled in at request time. This is deliberately shorter than a production prompt; it shows the blocks, not the polish.
>
> ```python
> PROMPT = """
> You answer questions about {domain} using only the passages below.
> Cite the passages you rely on by id, like [1]. If the answer is not in
> them, say you don't know rather than guessing.
>
> <passages>
> {passages}
> </passages>
>
> <question>
> {query}
> </question>
>
> Answer in {format}, and base every claim on the passages above.
> """
> ```
>
> The parts map onto the checklist above: task and scope in the first line, the retrieved context in a tagged block, the question last, and a closing grounding reminder so the key instruction is the last thing the model reads.

> [!warning] A fluent answer is not a grounded one
> A language model generates equally fluent text whether or not the facts are in its context, so fluency is no evidence of correctness. A generator must therefore be paired with checks that each claim it makes is supported by a retrieved passage, and the system should prefer saying "I don't know" to emitting an unsupported but plausible answer. Enforcing this across the whole system is the job of guardrails, covered in [[sec-rag-agentic]].

## The Synthesizer

A synthesizer keeps the generator's fluency and adds iteration. For questions that need evidence assembled across several steps, one retrieval pass is not enough, so the synthesizer loops: retrieve, read, decide what is still missing, form a follow-up query, and retrieve again. This is the multi-hop loop from [[sec-rag-pipeline]], and it is an early, hard-wired version of the agent loop in the next section.

A question such as "where do the 2025 Nobel laureates in physics work today?" shows the pattern. The answer lies in each laureate's own biography, but the question names no laureate, so a single retrieval pass returns what its words point at: prize lists and statistics, pages about the prize rather than about the people. The synthesizer's first hop retrieves passages about the prize and extracts the laureates' names. The second hop turns each name into a new query for that person's career. A generator then combines the hops, reconciles any contradictions, and summarizes them into one coherent answer. On the Nobel knowledge base of the chapter's demos, with the same budget of nine passages for both, a single pass reaches only one of the laureates' institutions, Santa Barbara, while the two hops reach Berkeley, Yale, Santa Barbara, Saclay, and Google. Hops pay off when the question does not name what the answer is about. The payoff is completeness on hard questions; the cost is latency and complexity, since each hop is another round trip through retrieval and reading.

## Citations

The ability to cite sources is what most clearly separates a RAG answer from a bare language-model answer. By linking each claim to the passage it came from, a citation-aware system becomes transparent and checkable, which is the foundation of user trust.

Citations are preserved through the pipeline rather than added at the end. During retrieval, each passage keeps its metadata: a passage identifier for internal tracking, the source document, its URL or location, the retrieval score, and the original text. As the answer is produced, the system records which passages contributed, creating an auditable link between answer and evidence. The final output pairs the answer with structured citation details:

```json
{
  "answer": "John Clarke, Michel H. Devoret, and John M. Martinis",
  "confidence": 0.95,
  "citation": {
    "passage_id": "passage_001",
    "source": "Nobel Prize Official Website",
    "url": "https://nobelprize.org/prizes/physics/2025/",
    "original_text": "The 2025 Nobel Prize in Physics was awarded to John Clarke, Michel H. Devoret, and John M. Martinis for their discovery of macroscopic quantum mechanical tunnelling and energy quantization in an electric circuit.",
    "retrieved_date": "2025-10-07"
  }
}
```

This structure pays off three ways. It lets users verify an answer by following the citation to the original material. It builds trust, because answers are visibly grounded in evidence rather than asserted. And it gives developers a trace to debug and refine the retrieval and reading stages, so the system improves over time rather than failing silently.

> [!demo] Reader, generator, and synthesizer {notebook="https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-03-result-generation.ipynb"}
> Run all three forms on the same retrieved passages: a BERT reader that marks the answer span, a generator with a grounded prompt, and a two-hop synthesizer against a single pass with the same budget, then check every citation quote against its passage.
>
> *Includes pre-run results: you can read through or download and experiment.*
