"""
data_ingestion/knowledge_extractor.py
=======================================
Uses a local LLM to extract Knowledge Graph triples 
(Entity 1, Relationship, Entity 2) from plain text chunks.
"""

import json
import re
import random
from loguru import logger
from llm_generation.llm_interface import BaseLLM

class KnowledgeExtractor:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
        self.system_prompt = (
            "You are a strict data extraction API. "
            "You MUST output exactly ONE valid JSON array of objects. "
            "Each object MUST have exactly three keys: 'source', 'target', and 'relation'. "
            "Do NOT output any markdown, do NOT output explanations, do NOT output conversational text. "
            "ONLY output the JSON. Example output:\n"
            '[{"source": "Apple", "target": "Tim Cook", "relation": "CEO"}]'
        )
        
    def _extract_json_array(self, text: str) -> str:
        """Attempt to find a JSON array block within conversational text."""
        # Find the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return "[]"
        
    def extract_from_chunk(self, chunk: str) -> list[dict]:
        """
        Extracts relationships from a single text chunk.
        Returns a list of dicts: [{"source": "A", "target": "B", "relation": "C"}]
        """
        user_prompt = f"Extract graph triples from this exact text block:\n\n{chunk}\n\nOUTPUT ONLY VALID JSON:"
        
        try:
            response_text = self.llm.generate(
                prompt=user_prompt,
                model=self.llm.name if hasattr(self.llm, "name") else "mistral",
                system=self.system_prompt,
                temperature=0.0
            )
            
            cleaned = self._extract_json_array(response_text)
                
            triples = json.loads(cleaned)
            if isinstance(triples, list):
                valid_triples = []
                for t in triples:
                    if "source" in t and "target" in t and "relation" in t:
                        src = str(t["source"]).strip().lower()
                        tgt = str(t["target"]).strip().lower()
                        rel = str(t["relation"]).strip().lower()
                        
                        # Very basic validation to avoid massive sentences acting as nodes
                        if len(src) < 50 and len(tgt) < 50:
                            valid_triples.append({
                                "source": src,
                                "target": tgt,
                                "relation": rel,
                                "context": chunk[:150]
                            })
                return valid_triples
            else:
                return []
                
        except json.JSONDecodeError:
            logger.warning(f"LLM json extraction failed. Raw output preview: {response_text[:100]}...")
            return []
        except Exception as e:
            logger.error(f"Knowledge extraction error: {e}")
            return []
            
    def process_chunks(self, chunks: list[str], max_chunks: int = 50) -> list[dict]:
        """Processes a sample of chunks to prevent massive wait times."""
        all_triples = []
        
        # If we have thousands of chunks, randomly sample max_chunks to extract the core graph
        # without taking 5 hours to run locally on a CPU.
        sample_chunks = chunks
        if len(chunks) > max_chunks:
            logger.info(f"Graph Extraction: Sampling {max_chunks} chunks out of {len(chunks)} for performance.")
            sample_chunks = random.sample(chunks, max_chunks)
            
        for i, chunk in enumerate(sample_chunks):
            # Skip chunks that are too tiny to have meaning
            if len(chunk.split()) < 10:
                 continue
                 
            logger.info(f"Extracting graph triples from chunk {i+1}/{len(sample_chunks)}...")
            triples = self.extract_from_chunk(chunk)
            all_triples.extend(triples)
            
        logger.info(f"Graph Extraction Complete: Discovered {len(all_triples)} relationships.")
        return all_triples
