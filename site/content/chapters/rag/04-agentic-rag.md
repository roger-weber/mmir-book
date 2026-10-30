# Agentic RAG {#sec-rag-agentic}

Retrieval-augmented generation has turned out to be one of the most useful patterns in applied AI, and enterprises have adopted it quickly. It combines the recall of a retrieval system with the fluency of a generator, keeps answers grounded in sources the organization controls, and needs no training or fine-tuning to absorb new or private information. A document added to the index this morning can be cited in an answer this afternoon.

What it has not removed is the pipeline. Everything so far is wired by hand into a fixed sequence, and that sequence encodes a single intent. A question-answering flow, a summarization flow, and a comparison flow are three different pipelines, each stitched together in code. A framework such as the [[sec-rag-frameworks|Haystack pipeline]] of the first section takes away much of the wiring, but the shape of the problem remains: one pipeline per intent, and every change is a code change that must be deployed. Worse, anything ad hoc falls outside the design. Answering "who won the 2025 Nobel Prize in Physics, and where does each laureate work?" is not a single retrieval; it is one lookup for the winners, then a further lookup per winner for their affiliation. A fixed pipeline cannot express that chain unless someone anticipated it and coded it, and the alternative, making the user ask each follow-up by hand, defeats the point of a direct answer.

## The Agent Loop

Agentic RAG drops the fixed sequence and gives control to the model. Instead of executing stages in a predetermined order, an agent runs a loop: it reasons about what the request needs, acts by calling a tool, observes the result, and decides whether it is done or must go round again. [[fig-agent-loop]] shows the shape.

![[fig-agent-loop]]

The three-step cycle, reason then act then observe, is the ReAct pattern, and the concrete mechanism behind the act step is tool calling, also known as function calling: the model emits a structured request naming a tool and its arguments, the system runs the tool, and the result returns as the next observation. The agent chooses each action as it goes, so the number of steps and the tools used are decided by the query rather than fixed in advance. A stop condition ends the loop, either when the agent judges it has enough to answer or when an iteration budget is reached, the latter a guard against a loop that never terminates.

Because the control flow is decided as it runs, the same loop serves both a known, repeatable flow and a one-off ad-hoc request. It also lets the agent recover from trouble that would halt a fixed pipeline. If the knowledge base is unavailable, it can try another source; if the evidence gathered so far is incomplete, it can fall back to a web search rather than return a half-answer.

## Walking the Loop: the 2025 Physics Nobel

To see the loop earn its keep, take the multi-part question from above and give the agent two tools: a Wikipedia knowledge base that returns article text, and a people-search tool that returns structured JSON about a person. A third tool, web search, is available as a fallback.

> Who won the 2025 Nobel Prize in Physics, and which universities are they affiliated with?

> [!example] the agent loop on the Nobel question
> 1. **Intent detection (reason).** The agent reads the question and recognizes two things: it asks for current facts it cannot answer from memory, and it has two parts, the winners and their affiliations. No separate classifier is needed; the intent is inferred inside the loop. Had the question been vague, such as "tell me about the physics Nobel", the agent could ask a clarifying question or infer the intent from the conversation so far, using meaning rather than keyword rules.
> 2. **Query transformation (act).** It calls the Wikipedia tool with a well-formed query, `"2025 Nobel Prize in Physics"`, rather than passing the user's raw sentence. The result it observes is an article naming John Clarke, Michel Devoret, and John Martinis.
> 3. **Search expansion (reason, then act again).** The affiliations are still missing, so the agent loops. For each laureate it calls people-search, typically with one call per name in a single step, and the directory returns structured data such as `{"name": "John Clarke", "affiliation": "University of California, Berkeley"}`. This per-winner chain is exactly the multi-hop flow a fixed pipeline could not express. The directory, like most real ones, is incomplete: it has no record for John Martinis, so for him the agent tries another way, a different spelling, the knowledge base, or, as a last resort, web search.
> 4. **Ad-hoc flow.** Nobody coded a winners-then-affiliations pipeline; the agent assembled it from the question. A flow that recurs often could instead be handed to the agent as a fixed skill, but it does not have to be.
> 5. **Final answer.** When the agent observes that the evidence is complete, it stops calling tools and generates a grounded, cited answer naming the three laureates and their universities, with the source of each fact.

The shift is from a Haystack-style pipeline hardcoded for one intent to a dynamic flow the agent assembles from a description of the tools it may use. We stop writing the sequence and start declaring the capabilities, and intent detection, once a stage of its own (see [[sec-advanced-text-processing-intent-routing|Advanced Text Processing]]), becomes simply the loop's first thought.

## Implementing the Loop

The loop itself is short. What makes it work is the tool declaration: each tool has a name, a description, and its parameters, and the description is what the model reads when it decides which tool to call. Declaring the tools is how we tell the agent what it may do.

```python
def schema(arg):
    """JSON Schema for a tool with one required string argument."""
    return {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}

TOOLS = [
    {"name": "wikipedia_search",
     "description": "Search a Wikipedia knowledge base about the Nobel Prizes and return "
                    "the text of the 3 best matching passages with their source.",
     "input_schema": schema("query")},
    {"name": "people_search",
     "description": "Look up a person in the people directory. Returns JSON with name, "
                    "affiliation, and position, or an error if there is no record.",
     "input_schema": schema("name")},
    {"name": "web_search",
     "description": "Search the web. Returns titles and short snippets. "
                    "Use only when the other tools return nothing.",
     "input_schema": schema("query")},
]

SYSTEM = ("Answer with the tools provided. Base every statement on tool results "
          "and name the source of each fact. If the tools do not give the answer, say so.")

def run_agent(question, max_steps=8):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):                                  # iteration budget
        reply = llm(messages, tools=TOOLS, system=SYSTEM)       # reason
        messages.append(reply.message)
        if not reply.tool_calls:                                # no further action: done
            return reply.text
        results = [tool_result(call.id, call_tool(call.name, call.arguments))
                   for call in reply.tool_calls]                # act
        messages.append({"role": "user", "content": results})  # observe
    return "I could not complete the answer within the step budget."
```

Each tool is declared with a JSON Schema of its arguments, the format the model provider expects; the system prompt sets the grounding rule for the whole run. The function `llm` sends the conversation and the declarations to the model, here Claude, and returns its reply with any tool calls it requests; `call_tool` runs the named tool; and `tool_result` wraps the output as an observation that refers back to the call it answers. Each pass through the loop is one reason-act-observe cycle: the model either requests tool calls or replies without one, which ends the loop. One reason step can request several calls at once, such as one people-search per laureate, so all their results return together in a single message. The growing message list is the agent's working memory, so in the Nobel example the laureates' names from the first tool call are in view when the agent decides to look up their affiliations. The `max_steps` budget is the stop condition that keeps a confused agent from looping forever.

A recurring flow can be handed to the same loop as a skill, a short description of the steps to follow, placed in the system prompt or loaded when the intent calls for it; the loop is unchanged. Frameworks package this pattern, including Haystack's agent component and graph-based libraries such as LangGraph, adding conveniences such as tool schemas generated from Python functions and tracing of each step. The guardrails below hook into the same two places in the loop: before `call_tool` runs, and before the final `return`.

## Guardrails

Handing control to a model raises the stakes, because the model now decides what to search, which tool to call, and what to answer. Guardrails are the checks that keep those decisions safe and grounded, and they sit at the two edges of the loop, as the dashed boxes in [[fig-agent-loop]] show.

Input guardrails act before the agent takes any action. They reject queries that are unsafe or outside the system's scope, so the agent neither wastes a tool call nor answers a question it has no business answering. A support assistant scoped to one insurance product, asked about a competitor's policy, should decline cleanly rather than improvise.

Output guardrails act before the answer reaches the user. They check two things: that the answer still serves the original intent and has not drifted over the course of the loop, and that every claim is supported by a retrieved result. An answer that goes beyond the evidence is a likely hallucination and must be caught; when a check fails, the safe response is to fall back to uncertainty rather than emit a confident but unsupported answer.

> [!warning] Guardrails run at every hop, not once
> In a fixed pipeline it is tempting to check the output only at the end. In an agent loop the model acts repeatedly, so an unsafe tool call or a drifting intermediate result can derail the steps that follow. Apply the input check before each action and the grounding check on each intermediate result, not only on the final answer.

The check before each action needs no model; it is plain code. Before `call_tool` runs, the loop verifies that the request names a declared tool, passes exactly the declared arguments, and leaves none of them empty. A blocked call is not silently dropped: the reason it was blocked becomes the observation, so the model sees the refusal and can choose another action.

## Key Findings

Step back and the significance is hard to miss. A single generic loop, given the right data and tools, can answer the whole range of queries that previously demanded a separate pipeline each. The agent supplies the control flow; the developer supplies the tools and the data behind them.

This is the engine behind the current wave of agentic assistants. Personal AI systems such as OpenClaw and Hermes push the idea to its limit, running one loop for everything a user might ask, in the same spirit as Claude Code. The retrieval components of this book, chunking, embeddings, hybrid retrieval, reranking, and reading, are all still present, but they have become tools the agent may call rather than stages a programmer must wire.

The deeper change is in who builds these systems and how. For the first time, assembling a retrieval system is less a programming task than a matter of defining the process and integrating the tools, work that increasingly happens in low-code or no-code environments. Describing what the agent may do, and trusting the loop to decide when to do it, has proven a more flexible way to build than wiring the stages by hand.

> [!demo] Agentic RAG {notebook="https://github.com/roger-weber/mmir-unibasel-hs26/blob/main/demos/ch07-04-agentic-rag.ipynb"}
> Run the reason-act-observe loop with Claude and three tools on the Nobel question, read its trajectory step by step, see the same loop answer questions of different shapes, cut it short with the step budget, and add scope, tool-call, and grounding guardrails (recorded runs replay without AWS access).
>
> *Includes pre-run results: you can read through or download and experiment.*
