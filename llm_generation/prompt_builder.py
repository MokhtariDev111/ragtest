"""
llm_generation/prompt_builder.py
==================================
Constructs system and user prompts from retrieved context chunks.

Supports English and French prompt templates.
"""

from __future__ import annotations

from loguru import logger


_PROMPT_TEMPLATES = {
    "en": {
        "system": (
            "You are a helpful AI assistant. Answer the user's question "
            "based ONLY on the provided context. If the context does not "
            "contain enough information, say so clearly. Do not hallucinate."
        ),
        "user": (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    },
    "fr": {
        "system": (
            "Vous êtes un assistant IA utile. Répondez à la question de "
            "l'utilisateur en vous basant UNIQUEMENT sur le contexte fourni. "
            "Si le contexte ne contient pas assez d'informations, dites-le "
            "clairement. Ne fabricquez pas d'informations."
        ),
        "user": (
            "Contexte :\n{context}\n\n"
            "Question : {question}\n\n"
            "Réponse :"
        ),
    },
}


class PromptBuilder:
    """
    Builds a full prompt string from retrieved context chunks and a question.

    Parameters
    ----------
    language        : 'en' | 'fr'
    max_context_len : maximum character length of the concatenated context
    include_system  : whether to prepend the system instruction to the prompt
    """

    def __init__(
        self,
        language: str = "en",
        max_context_len: int = 3000,
        include_system: bool = True,
    ):
        lang = language if language in _PROMPT_TEMPLATES else "en"
        self.templates = _PROMPT_TEMPLATES[lang]
        self.max_context_len = max_context_len
        self.include_system = include_system
        if language not in _PROMPT_TEMPLATES:
            logger.warning(f"Language '{language}' not supported, defaulting to 'en'")

    def build(self, question: str, context_chunks: list[str]) -> str:
        """
        Assemble a complete prompt string.

        Parameters
        ----------
        question       : the user's question
        context_chunks : list of retrieved text chunks (ordered by relevance)

        Returns
        -------
        str – the complete prompt ready to send to the LLM
        """
        # Concatenate chunks, trim to max_context_len
        context = "\n\n---\n\n".join(context_chunks)
        if len(context) > self.max_context_len:
            context = context[: self.max_context_len] + "\n...[truncated]"

        user_prompt = self.templates["user"].format(
            context=context,
            question=question,
        )

        if self.include_system:
            return self.templates["system"] + "\n\n" + user_prompt

        return user_prompt

    def build_from_str(self, question: str, context: str) -> str:
        """Build a prompt from a single pre-assembled context string."""
        if len(context) > self.max_context_len:
            context = context[: self.max_context_len] + "\n...[truncated]"
        user_prompt = self.templates["user"].format(
            context=context,
            question=question,
        )
        if self.include_system:
            return self.templates["system"] + "\n\n" + user_prompt
        return user_prompt
