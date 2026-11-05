"""Grounded answer generation with citations and an explicit 'not in the documents' path."""
from pydantic import BaseModel, Field

from .retrieval import Hit

ANSWER_PROMPT = """You answer questions about Lumen Health Assurance policies for members and claims staff.

Use ONLY the numbered sources below. Rules:
- Every factual sentence must cite its source(s) like [S2].
- Quote amounts, limits and time periods exactly as the source states them.
- If the answer depends on the plan and the question does not say which plan, give the answer for each plan
  that the sources cover.
- If the sources do not contain the answer, set answerable=false and say you cannot find it in the policy
  documents. Do not use outside knowledge and do not guess.

<sources>
{sources}
</sources>

Question: {question}"""


class Answer(BaseModel):
    answerable: bool = Field(description="False if the sources do not contain the answer.")
    answer: str = Field(description="Concise answer with [S#] citations.")
    cited_sources: list[str] = Field(description="Source labels used, e.g. ['S1', 'S3'].")


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[S{i}] {h.text}" for i, h in enumerate(hits, start=1))


def generate(llm, question: str, hits: list[Hit]) -> Answer:
    if not hits:
        return Answer(answerable=False, answer="I cannot find this in the policy documents.", cited_sources=[])
    return llm.with_structured_output(Answer).invoke(ANSWER_PROMPT.format(sources=format_sources(hits), question=question))
