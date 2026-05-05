from nullwatch import NullwatchClient, Span, Eval
from nullwatch.scorers import RAGHallucinationScorer

# Mock RAG pipeline
CONTEXT_DOCS = [
    "France is a country in Western Europe. "
    "The capital of France is Paris. "
    "The population of France is approximately 68 million people.",
    "The Eiffel Tower is located in Paris and was built in 1889. "
    "It was designed by Gustave Eiffel for the World's Fair.",
]

QUESTION = "What is the capital of France and when was the Eiffel Tower built?"

# Grounded answer (should pass)
ANSWER_CLEAN = "The capital of France is Paris. The Eiffel Tower was built in 1889."

# Hallucinated answer (should fail — wrong population and year)
ANSWER_HALLUCINATED = (
    "The capital of France is Paris. "
    "The population of France is 80 million. "
    "The Eiffel Tower was built in 1901 by Napoleon."
)

# Setup
client = NullwatchClient(raise_on_error=False)
scorer = RAGHallucinationScorer(dataset="demo-rag")

RUN_ID = "run-rag-demo-001"

# Process clean answer
print("=" * 60)
print("Testing CLEAN answer:")
print(f"  Answer: {ANSWER_CLEAN}")

with client.span(RUN_ID, "llm.call", model="gpt-4o") as s:
    # In a real pipeline, you'd call your LLM here
    answer = ANSWER_CLEAN
    s.input_tokens = 300
    s.output_tokens = 30

# Score hallucination
eval_clean = scorer.score(
    run_id=RUN_ID,
    contexts=CONTEXT_DOCS,
    question=QUESTION,
    answer=answer,
)
client.ingest_eval(eval_clean)

print(f"  Verdict: {eval_clean.verdict}")
print(f"  Score:   {eval_clean.score:.3f}")
print(f"  Notes:   {eval_clean.notes}")

# Process hallucinated answer
print()
print("=" * 60)
print("Testing HALLUCINATED answer:")
print(f"  Answer: {ANSWER_HALLUCINATED}")

with client.span(RUN_ID, "llm.call", model="gpt-4o") as s:
    answer = ANSWER_HALLUCINATED
    s.input_tokens = 300
    s.output_tokens = 45

eval_hallucinated = scorer.score(
    run_id=RUN_ID,
    contexts=CONTEXT_DOCS,
    question=QUESTION,
    answer=answer,
)
client.ingest_eval(eval_hallucinated)

print(f"  Verdict: {eval_hallucinated.verdict}")
print(f"  Score:   {eval_hallucinated.score:.3f}")
print(f"  Notes:   {eval_hallucinated.notes}")

# Fetch run summary
print()
print("=" * 60)
summary = client.get_run(RUN_ID)
if summary:
    print(f"Run summary:")
    print(f"  Spans:   {summary.span_count}")
    print(f"  Evals:   {summary.eval_count}")
    print(f"  Passed:  {summary.pass_count}")
    print(f"  Failed:  {summary.fail_count}")
    print(f"  Verdict: {summary.verdict}")
else:
    print("(nullwatch server not running — no run summary available)")
