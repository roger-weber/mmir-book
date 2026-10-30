# Evaluating RAG {#sec-rag-evaluation}

Every section of this chapter has added choices. We can chunk by size, by sentence, by structure, or by meaning; embed with one model or another; weight BM25 against dense retrieval; rerank or not; retrieve five passages or twenty; pick a generator and a prompt; and let an agent loop for two steps or ten. Classical retrieval asked how well one ranking function performs on a benchmark. A RAG system forces a broader question: of all these combinations, which one serves the users best, and how would we know?

Precision and recall are part of the answer but no longer all of it. Three further dimensions enter: the quality of the generated answer, the time it takes to produce, and what it costs. The table below shows how the main groups of choices map onto what they mainly affect.

| Choice | Example options | Mainly affects |
|---|---|---|
| Document processing | chunking strategy, chunk size, overlap | retrieval recall, index size |
| Retrieval | embedding model, BM25 weight, top-$k$ | recall, precision, retrieval latency |
| Reranking | reranker model, candidates reranked | precision, latency |
| Generation | generator model, prompt, context size | answer quality, token cost |
| Agent control | tools available, iteration limit | answer quality, number of tool calls, total latency |

## Retrieval Still Sets the Ceiling

A generated answer can never be better than the evidence it is built from, so the retrieval metrics of [[sec-performance-evaluation-precision-recall|Performance Evaluation]] still apply, recast to describe the retrieved context rather than a result list. Context recall asks whether everything needed to answer the question was retrieved. If a required fact never reaches the generator, no model and no prompt can recover it; the best it can do is admit the gap, and the worst is to invent the missing piece. Context precision asks how much of the retrieved context is actually relevant. Irrelevant passages are not harmless padding: they compete for the generator's attention, and closely related but wrong evidence is the most dangerous kind.

> [!example] context recall and precision on the Nobel question
> Recall the running question: "Who won the 2025 Nobel Prize in Physics, and which universities are they affiliated with?" A complete answer needs four pieces of evidence: one passage naming the three laureates and one affiliation for each of them.
>
> A fixed pipeline that retrieves the top five chunks for the whole question returns the passage naming the winners, one laureate's affiliation, two chunks about the 2025 Nobel Prize in Chemistry, and one about the award ceremony. Two of the four required pieces arrived, so context recall is $2/4 = 0.5$; two of the five chunks are relevant, so context precision is $2/5 = 0.4$. The answer will be missing two affiliations, and the Chemistry chunks invite the generator to attach a chemist's university to a physicist.
>
> The agent from [[sec-rag-agentic]] first retrieves the winners and then looks up each laureate in turn. It gathers all four pieces and nothing else: context recall and context precision are both $1$.

Recall needs to know the full set of evidence an answer requires, so it depends on a labelled reference. Precision over the retrieved context can often be judged without one, by asking whether each passage bears on the question. Both trade off as they do in classical retrieval: retrieving more raises the chance of capturing every required passage but dilutes the context with noise.

## The Price of an Answer

A ranked list arrives in milliseconds and costs a fraction of a cent. A generated answer can take seconds, and every request consumes tokens that someone pays for. An agent multiplies both, since each pass through the loop is another model call and possibly another tool call. The bar is therefore higher than for classical search: a RAG answer must be clearly better for the user, not merely different, to justify the extra latency and cost.

Latency should be measured as a high percentile rather than an average. The 95th percentile, or p95, is the time within which 95 per cent of requests complete, and it reflects the slow cases users actually notice. Cost is best expressed per answered question, combining input tokens, output tokens, and tool charges.

For an agent, the path to the answer matters as well as the answer itself. A correct answer reached in two tool calls is better than the same answer reached in ten, even though both are equally correct. Evaluating the trajectory means recording how many iterations the loop took, which tools it called, whether each call was the right one, and whether it kept searching after it already had enough evidence.

## Measuring Answer Quality

The hardest dimension to measure is the quality of the answer itself, because it is free text rather than a ranked list. It is useful to separate four questions, which overlap but are not interchangeable.

- **Correctness**: is the answer factually right? This is judged against a trusted reference answer or a list of expected facts.
- **Faithfulness**: is every claim in the answer supported by the retrieved context? This measures grounding.
- **Relevance**: does the answer address the question that was asked, rather than a neighbouring one?
- **Completeness**: does the answer contain all the facts the question requires?

Faithfulness is the dimension specific to RAG, since it ties the answer to the evidence the system actually retrieved.

> [!formula] Faithfulness {#eq-faithfulness}
> $$\text{faithfulness} = \frac{\lvert \{\text{claims in the answer supported by the context}\} \rvert}{\lvert \{\text{claims in the answer}\} \rvert}$$
>
> The fraction of the answer's claims that the retrieved passages support. A value of 1 means every claim is grounded; a lower value means the generator added content the evidence does not back.

> [!example]
> The answer reads: "The 2025 Nobel Prize in Physics went to Clarke, Devoret, and Martinis, who shared 11 million Swedish kronor." The retrieved context names the three laureates but says nothing about the prize money. Of the two claims, one is supported, so faithfulness is $1/2 = 0.5$. The prize amount happens to be correct, yet it still lowers faithfulness: the generator took it from its own memory rather than from the evidence, and a claim taken from memory can just as easily be wrong.

> [!warning] Faithful is not the same as correct
> Faithfulness measures agreement with the retrieved context, not with the world. A true claim that the context does not support lowers faithfulness, as in the example above. And a perfectly faithful answer can be wrong if the retrieved documents are outdated or mistaken. Measure correctness separately, against a reference.

The four dimensions of answer quality, correctness, faithfulness, relevance, and completeness, are measured in two modes. Reference-based evaluation compares the answer with a trusted expected answer; it gives the strongest signal of correctness and completeness but requires someone to write the references. Reference-free evaluation judges faithfulness and relevance from the question, the answer, and the retrieved context alone, which scales to thousands of production requests where no reference exists. A sound evaluation uses both: a smaller reference set for correctness and a larger reference-free sweep for grounding.

Most of these judgements are made by another language model acting as an evaluator, an approach called LLM-as-a-judge. For faithfulness, the judge splits the answer into individual claims and checks each against the context. RAGAS packages the common metrics in this way: context precision and recall, faithfulness, and answer relevance, along with reference-based correctness.

> [!warning] Validate the judge before trusting it
> An LLM judge is itself a model, with its own blind spots: it can miss subtle factual errors, favour longer answers, and change its verdicts when the evaluation prompt changes. Check its scores against a sample of human judgements before relying on them, and keep the judge model and prompt fixed while comparing systems. Where a rule can decide, such as a date, an amount, or whether a cited passage exists, use a deterministic check instead of a judge.

Measuring retrieval and answer quality separately also tells us where to look when something goes wrong.

| Retrieval quality | Answer quality | Likely diagnosis |
|---|---|---|
| Low | Low | Fix retrieval: chunking, indexing, or query formulation. |
| High | Low | Fix generation: the prompt, the model, or how the context is used. |
| Low | High | Suspicious: the model may be answering from memory rather than evidence. |
| High | High | Working; look for savings in latency and cost. |

These are heuristics rather than proofs, but they turn a single disappointing score into a direction to investigate.

## Comparing Two Configurations

The practical question is usually narrower than "which configuration is best". We have a system A and a proposed change B, perhaps a new chunking strategy, a reranker, or an agent loop, and we want to know whether B is better. The method is a controlled experiment.

Start from a fixed benchmark: a set of representative questions, each with a reference answer and the evidence it requires, built the same way as the test collections of [[sec-performance-evaluation-benchmark-design]]. Run A and B on exactly the same questions, changing only the one component under test, and record for every question the retrieved context, the answer, the latency, the tokens used, and for an agent the tool calls. Then compute every dimension for both systems. The sketch below shows the shape of such a harness; the `judge_*` functions stand for evaluators such as those RAGAS provides.

```python
def evaluate(system, benchmark):
    rows = []
    for item in benchmark:   # question, reference answer, required evidence
        start = time.perf_counter()
        run = system.answer(item.question)   # answer, retrieved, tool_calls, tokens
        latency = time.perf_counter() - start   # stop the clock before evaluating
        rows.append({
            "context_recall": recall(run.retrieved, item.evidence),
            "context_precision": precision(run.retrieved, item.evidence),
            "correct": judge_correct(run.answer, item.reference),
            "faithfulness": judge_faithful(run.answer, run.retrieved),
            "latency_s": latency,
            "tokens": run.tokens,
            "tool_calls": len(run.tool_calls),
        })
    return rows

results_a = evaluate(system_a, benchmark)
results_b = evaluate(system_b, benchmark)
```

A higher average for B is not yet proof that B is better. Benchmarks are small, and some questions are easy or hard for reasons unrelated to the change. Because both systems answered the same questions, the right comparison is paired: look at the questions where the two systems disagree, as in the paired tests of [[sec-performance-evaluation-ranked-results]].

> [!example] is B really better?
> On a benchmark of 50 questions, system A answers 38 correctly (76 per cent) and system B answers 44 (88 per cent). Is B better, or did it get lucky? The test proceeds in four steps.
>
> 1. **Keep only the disagreements.** Both systems get 36 of the same questions right and 4 of the same questions wrong. These 40 questions say nothing about which system is better, so we set them aside. What remains are the 10 questions where the systems disagree: B is right and A wrong on 8, A is right and B wrong on 2.
> 2. **Assume there is no difference.** If B were no better than A, each disagreement would be equally likely to go either way, like the toss of a fair coin. The number of disagreements won by B would then follow a binomial distribution: the probability that B wins exactly $k$ of the 10 is $\binom{10}{k} / 2^{10}$, where $\binom{10}{k}$ counts the ways to choose which $k$ of the 10 questions B wins.
> 3. **Measure how extreme the observed split is.** The probability of exactly 8 to 2 is $45/1024 \approx 0.044$, but that number is misleading on its own. Every individual outcome is unlikely, even the most typical one: an even 5 to 5 split has probability $252/1024 \approx 0.25$, and the probability of any single exact outcome shrinks further as the number of disagreements grows. The meaningful question is how often chance alone produces a result at least as one-sided as the one we saw, so we add up 8 to 2, 9 to 1, and 10 to 0: $(45 + 10 + 1)/1024 \approx 0.055$.
> 4. **Count both directions.** Before the experiment we did not know which system would come out ahead, and a 2 to 8 split in favour of A would have been just as surprising. Since the distribution is symmetric, we double the probability:
>
> $$p = 2 \cdot \frac{\binom{10}{8} + \binom{10}{9} + \binom{10}{10}}{2^{10}} = 2 \cdot \frac{45 + 10 + 1}{1024} \approx 0.11.$$
>
> If A and B were truly equally good, a split at least this uneven would still occur about one time in nine. That is not rare enough to rule out chance, against the common threshold of $0.05$, so the twelve-point gap is not conclusive. With the same pattern on a benchmark of 100 questions, B winning 16 of 20 disagreements, the same calculation gives $p \approx 0.012$, and the improvement becomes convincing. The ratio is unchanged; what changed is the amount of evidence behind it.

Quality is only half of the decision. If B is more accurate but doubles the p95 latency or the cost per answer, whether it is better depends on what the application can afford, and that is a product decision the measurements inform rather than settle. A good habit is to fix the acceptable latency and cost before running the experiment, so the result cannot quietly move the goalposts.

Tooling helps with the bookkeeping. Metric libraries such as RAGAS and DeepEval supply the judges and turn quality thresholds into regression tests, and tracing tools such as Phoenix and LangSmith record each run so that a surprising score can be traced to the step that caused it. Running such comparisons systematically over many parameters at once, known as hyperparameter search, builds on exactly this A-versus-B foundation and goes beyond the scope of this book.

## When Not to Use RAG

The cost argument above cuts both ways. RAG is not the default answer to every question-answering problem, and reaching for it when a simpler tool fits adds cost and latency for no benefit. Four situations call for a different approach.

- **The knowledge is stable and general.** The model already holds it. Asking a current model for a textbook definition or a historical fact gains nothing from retrieval and only adds a retrieval step that can fail.
- **The corpus fits in the context window.** When the entire relevant collection is small enough, long-context prompting can be simpler and more reliable than retrieval, because there is no chance of retrieving the wrong passage. As context windows grow, the size threshold for this keeps rising.
- **The task is to change behaviour, not to supply facts.** When the goal is a different style, tone, or output format, fine-tuning is the right tool, since RAG changes what the model reads, not how it writes.
- **The question maps onto structured data.** For a query such as "how many orders shipped last week", a direct database or API query, possibly generated by a text-to-SQL model, gives an exact answer where retrieval over unstructured text would only approximate one.

An agent softens the choice, because it can decide per query: it can answer a stable fact from memory, send a structured question to a database tool, and reserve document retrieval for the questions that need it. The question then shifts from "RAG or not" to "which tool for this query".

Even where RAG is appropriate, it carries ongoing costs that the alternatives do not: an index to build and keep fresh, retrieval latency on every query, and a quality ceiling set by the corpus. If the document collection is noisy or out of date, retrieval faithfully supplies bad evidence, and the grounded answer is confidently wrong. RAG is the right choice when answers must draw on private, changing, or very specific knowledge and must be verifiable against sources, and the wrong choice when a cheaper tool already meets those needs.

> [!demo] Evaluating RAG {notebook="https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-05-rag-evaluation.ipynb"}
> Compare a fixed pipeline with an agent on a small Nobel benchmark: context recall and precision, p95 latency and tokens per answer, correctness and faithfulness with an LLM judge checked against a rule, and a paired sign test on the questions where the two disagree.
>
> *Includes pre-run results: you can read through or download and experiment.*
