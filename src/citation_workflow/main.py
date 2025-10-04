import asyncio
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from dotenv import load_dotenv
from openai import OpenAI

from bookwyrm import AsyncBookWyrmClient
from bookwyrm.models import (
    TextSpan, 
    TextSpanResult, 
    CitationStreamResponse, 
    CitationSummaryResponse,
    CitationProgressUpdate
)

# Load environment variables
load_dotenv()


class CharacterToPageMapper:
    """Maps character positions in concatenated text to original page numbers."""
    
    def __init__(self):
        self.page_boundaries: List[int] = []  # Character positions where pages end
        self.page_numbers: List[int] = []     # Corresponding page numbers
    
    def add_page(self, page_number: int, page_text: str, current_position: int):
        """Add a page's text and record its boundaries."""
        end_position = current_position + len(page_text)
        self.page_boundaries.append(end_position)
        self.page_numbers.append(page_number)
    
    def get_page_number(self, char_position: int) -> int:
        """Get the page number for a given character position."""
        for i, boundary in enumerate(self.page_boundaries):
            if char_position < boundary:
                return self.page_numbers[i]
        # If we're past all boundaries, return the last page
        return self.page_numbers[-1] if self.page_numbers else 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'page_boundaries': self.page_boundaries,
            'page_numbers': self.page_numbers
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CharacterToPageMapper':
        """Create from dictionary."""
        mapper = cls()
        mapper.page_boundaries = data['page_boundaries']
        mapper.page_numbers = data['page_numbers']
        return mapper


def get_cache_dir(pdf_path: str) -> Path:
    """Get cache directory for a PDF file."""
    pdf_path_obj = Path(pdf_path)
    # Create a hash of the PDF path to avoid filesystem issues
    path_hash = hashlib.md5(str(pdf_path_obj.absolute()).encode()).hexdigest()[:8]
    cache_dir = Path('.citation_cache') / f"{pdf_path_obj.stem}_{path_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache(pdf_path: str):
    """Clear all cache files for a PDF."""
    cache_dir = get_cache_dir(pdf_path)
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        click.echo(f"Cleared cache for {pdf_path}")


async def extract_pdf_text(pdf_path: str, api_key: Optional[str] = None, use_cache: bool = True) -> Tuple[str, CharacterToPageMapper]:
    """Extract text from PDF and create character-to-page mapping."""
    cache_dir = get_cache_dir(pdf_path)
    extraction_cache = cache_dir / "extraction.json"
    
    # Try to load from cache first
    if use_cache and extraction_cache.exists():
        click.echo(f"Loading PDF text from cache: {extraction_cache}")
        try:
            with open(extraction_cache, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            raw_text = cached_data['raw_text']
            mapper = CharacterToPageMapper.from_dict(cached_data['page_mapper'])
            click.echo(f"Loaded {len(raw_text)} characters from cache")
            return raw_text, mapper
        except Exception as e:
            click.echo(f"Cache read failed, extracting fresh: {e}")
    
    click.echo(f"Extracting text from PDF: {pdf_path}")
    
    async with AsyncBookWyrmClient(api_key=api_key) as client:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        response = await client.extract_pdf(pdf_bytes=pdf_bytes)
        
        raw_text = ""
        mapper = CharacterToPageMapper()
        
        for page in response.pages:
            page_text = ""
            # Concatenate all text blocks from the page
            for text_block in page.text_blocks:
                page_text += text_block.text + "\n"
            
            # Record page boundary before adding text
            mapper.add_page(page.page_number, page_text, len(raw_text))
            raw_text += page_text
        
        click.echo(f"Extracted {len(raw_text)} characters from {response.total_pages} pages")
        
        # Save to cache
        if use_cache:
            try:
                cache_data = {
                    'raw_text': raw_text,
                    'page_mapper': mapper.to_dict(),
                    'total_pages': response.total_pages
                }
                with open(extraction_cache, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                click.echo(f"Saved extraction to cache: {extraction_cache}")
            except Exception as e:
                click.echo(f"Failed to save extraction cache: {e}")
        
        return raw_text, mapper


async def process_text_to_chunks(text: str, pdf_path: str, api_key: Optional[str] = None, use_cache: bool = True) -> List[TextSpan]:
    """Process raw text into chunks using BookWyrm's phrasal model."""
    cache_dir = get_cache_dir(pdf_path)
    chunks_cache = cache_dir / "chunks.json"
    
    # Try to load from cache first
    if use_cache and chunks_cache.exists():
        click.echo(f"Loading text chunks from cache: {chunks_cache}")
        try:
            with open(chunks_cache, 'r', encoding='utf-8') as f:
                cached_chunks = json.load(f)
            chunks = [TextSpan(**chunk_data) for chunk_data in cached_chunks]
            click.echo(f"Loaded {len(chunks)} chunks from cache")
            return chunks
        except Exception as e:
            click.echo(f"Chunks cache read failed, processing fresh: {e}")
    
    click.echo("Processing text into chunks using phrasal model...")
    
    chunks = []
    async with AsyncBookWyrmClient(api_key=api_key) as client:
        async for response in client.stream_process_text(
            text=text,
            chunk_size=1000,  # Reasonable chunk size for citation analysis
            offsets=True
        ):
            if isinstance(response, TextSpanResult):
                chunks.append(TextSpan(
                    text=response.text,
                    start_char=response.start_char,
                    end_char=response.end_char
                ))
    
    click.echo(f"Created {len(chunks)} text chunks")
    
    # Save to cache
    if use_cache:
        try:
            chunks_data = [
                {
                    'text': chunk.text,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
                for chunk in chunks
            ]
            with open(chunks_cache, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            click.echo(f"Saved chunks to cache: {chunks_cache}")
        except Exception as e:
            click.echo(f"Failed to save chunks cache: {e}")
    
    return chunks


async def find_citations(chunks: List[TextSpan], query: str, pdf_path: str, api_key: Optional[str] = None, use_cache: bool = True) -> List[dict]:
    """Find citations related to the query event."""
    cache_dir = get_cache_dir(pdf_path)
    # Create a hash of the query to handle different queries for same PDF
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    citations_cache = cache_dir / f"citations_{query_hash}.json"
    
    # Try to load from cache first
    if use_cache and citations_cache.exists():
        click.echo(f"Loading citations from cache: {citations_cache}")
        try:
            with open(citations_cache, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            if cached_data.get('query') == query:  # Verify query matches
                citations = cached_data['citations']
                click.echo(f"Loaded {len(citations)} citations from cache")
                return citations
            else:
                click.echo("Query mismatch in cache, finding fresh citations")
        except Exception as e:
            click.echo(f"Citations cache read failed, finding fresh: {e}")
    
    click.echo(f"Finding citations for query: {query}")
    
    citations = []
    async with AsyncBookWyrmClient(api_key=api_key) as client:
        async for response in client.stream_citations(
            chunks=chunks,
            question=query
        ):
            if isinstance(response, CitationProgressUpdate):
                click.echo(f"Progress: {response.message}")
            elif isinstance(response, CitationStreamResponse):
                citation = response.citation
                citations.append({
                    'text': citation.text,
                    'reasoning': citation.reasoning,
                    'quality': citation.quality,
                    'start_chunk': citation.start_chunk,
                    'end_chunk': citation.end_chunk
                })
            elif isinstance(response, CitationSummaryResponse):
                click.echo(f"Found {response.total_citations} total citations")
    
    # Save to cache
    if use_cache:
        try:
            cache_data = {
                'query': query,
                'citations': citations,
                'total_citations': len(citations)
            }
            with open(citations_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            click.echo(f"Saved citations to cache: {citations_cache}")
        except Exception as e:
            click.echo(f"Failed to save citations cache: {e}")
    
    return citations


def score_citations_with_llm(citations: List[dict], query: str, pdf_path: str, openai_api_key: Optional[str] = None, use_cache: bool = True) -> List[dict]:
    """Score citations using OpenAI LLM (1-5 scale)."""
    cache_dir = get_cache_dir(pdf_path)
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    scores_cache = cache_dir / f"scores_{query_hash}.json"
    
    # Try to load from cache first
    scored_citations = []
    citations_to_score = []
    citation_index_map = {}
    
    if use_cache and scores_cache.exists():
        click.echo(f"Loading LLM scores from cache: {scores_cache}")
        try:
            with open(scores_cache, 'r', encoding='utf-8') as f:
                cached_scores = json.load(f)
            
            # Match citations with cached scores by text hash
            for i, citation in enumerate(citations):
                citation_hash = hashlib.md5(citation['text'].encode()).hexdigest()
                if citation_hash in cached_scores:
                    # Use cached score
                    citation_with_score = citation.copy()
                    citation_with_score.update(cached_scores[citation_hash])
                    scored_citations.append(citation_with_score)
                else:
                    # Need to score this one
                    citations_to_score.append(citation)
                    citation_index_map[len(citations_to_score) - 1] = len(scored_citations)
                    scored_citations.append(None)  # Placeholder
            
            if not citations_to_score:
                click.echo(f"All {len(citations)} citations loaded from cache")
                return [c for c in scored_citations if c is not None]
            else:
                click.echo(f"Loaded {len(citations) - len(citations_to_score)} scores from cache, need to score {len(citations_to_score)} more")
        except Exception as e:
            click.echo(f"Scores cache read failed, scoring all fresh: {e}")
            citations_to_score = citations
            scored_citations = []
            citation_index_map = {i: i for i in range(len(citations))}
    else:
        citations_to_score = citations
        scored_citations = [None] * len(citations)
        citation_index_map = {i: i for i in range(len(citations))}
    
    if not citations_to_score:
        return scored_citations
    
    click.echo(f"Scoring {len(citations_to_score)} citations with LLM...")
    
    client = OpenAI(api_key=openai_api_key)
    new_scores = {}
    
    for i, citation in enumerate(citations_to_score):
        click.echo(f"Scoring citation {i+1}/{len(citations_to_score)}")
        
        prompt = f"""
You are evaluating how well a text citation answers a specific query about an event.

Query: {query}

Citation text: {citation['text']}

Citation reasoning: {citation['reasoning']}

Please score this citation on a scale of 1-5:
1 = Unrelated to the query
2 = Tangentially related but not useful
3 = Somewhat related with minor relevance
4 = Clearly related and provides good evidence
5 = Comprehensive account that fully addresses the query

Respond with just the number (1-5) and a brief explanation.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Extract score (first digit found)
            score = None
            for char in llm_response:
                if char.isdigit() and 1 <= int(char) <= 5:
                    score = int(char)
                    break
            
            if score is None:
                score = 3  # Default score if parsing fails
                
            citation_with_score = citation.copy()
            citation_with_score['llm_score'] = score
            citation_with_score['llm_explanation'] = llm_response
            
            # Update the scored_citations list
            original_index = citation_index_map[i]
            scored_citations[original_index] = citation_with_score
            
            # Store for cache
            citation_hash = hashlib.md5(citation['text'].encode()).hexdigest()
            new_scores[citation_hash] = {
                'llm_score': score,
                'llm_explanation': llm_response
            }
            
        except Exception as e:
            click.echo(f"Error scoring citation: {e}")
            citation_with_score = citation.copy()
            citation_with_score['llm_score'] = 3  # Default score
            citation_with_score['llm_explanation'] = f"Error during scoring: {e}"
            
            original_index = citation_index_map[i]
            scored_citations[original_index] = citation_with_score
            
            citation_hash = hashlib.md5(citation['text'].encode()).hexdigest()
            new_scores[citation_hash] = {
                'llm_score': 3,
                'llm_explanation': f"Error during scoring: {e}"
            }
    
    # Save new scores to cache
    if use_cache and new_scores:
        try:
            # Load existing cache and merge
            existing_scores = {}
            if scores_cache.exists():
                with open(scores_cache, 'r', encoding='utf-8') as f:
                    existing_scores = json.load(f)
            
            existing_scores.update(new_scores)
            
            with open(scores_cache, 'w', encoding='utf-8') as f:
                json.dump(existing_scores, f, ensure_ascii=False, indent=2)
            click.echo(f"Saved {len(new_scores)} new scores to cache: {scores_cache}")
        except Exception as e:
            click.echo(f"Failed to save scores cache: {e}")
    
    return [c for c in scored_citations if c is not None]


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.argument('query_document', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file path')
@click.option('--bookwyrm-api-key', envvar='BOOKWYRM_API_KEY', help='BookWyrm API key')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--no-cache', is_flag=True, help='Disable caching (process everything fresh)')
@click.option('--clear-cache', is_flag=True, help='Clear existing cache before processing')
def main(pdf_path: str, query_document: str, output: Optional[str], 
         bookwyrm_api_key: Optional[str], openai_api_key: Optional[str],
         no_cache: bool, clear_cache: bool):
    """
    Citation workflow CLI.
    
    Extracts text from PDF, processes it with BookWyrm's phrasal model,
    finds citations related to events described in the query document,
    and scores them with an LLM.
    
    By default, intermediate results are cached to allow resumption if interrupted.
    Use --no-cache to disable caching or --clear-cache to start fresh.
    
    PDF_PATH: Path to the PDF file to analyze
    QUERY_DOCUMENT: Path to text file describing the event to search for
    """
    
    async def run_workflow():
        use_cache = not no_cache
        
        # Clear cache if requested
        if clear_cache:
            clear_cache(pdf_path)
        
        # Read query document
        with open(query_document, 'r', encoding='utf-8') as f:
            query = f.read().strip()
        
        click.echo(f"Query: {query}")
        if use_cache:
            cache_dir = get_cache_dir(pdf_path)
            click.echo(f"Cache directory: {cache_dir}")
        
        # Step 1: Extract PDF text with page mapping
        raw_text, page_mapper = await extract_pdf_text(pdf_path, bookwyrm_api_key, use_cache)
        
        # Step 2: Process text into chunks
        chunks = await process_text_to_chunks(raw_text, pdf_path, bookwyrm_api_key, use_cache)
        
        # Step 3: Find citations
        citations = await find_citations(chunks, query, pdf_path, bookwyrm_api_key, use_cache)
        
        # Step 4: Score citations with LLM
        scored_citations = score_citations_with_llm(citations, query, pdf_path, openai_api_key, use_cache)
        
        # Add page information to citations
        for citation in scored_citations:
            if citation['start_chunk'] < len(chunks):
                chunk = chunks[citation['start_chunk']]
                start_page = page_mapper.get_page_number(chunk.start_char)
                end_page = page_mapper.get_page_number(chunk.end_char)
                citation['start_page'] = start_page
                citation['end_page'] = end_page
        
        # Sort by LLM score (highest first)
        scored_citations.sort(key=lambda x: x['llm_score'], reverse=True)
        
        # Prepare results
        results = {
            'query': query,
            'pdf_path': pdf_path,
            'total_citations': len(scored_citations),
            'citations': scored_citations,
            'summary': {
                'score_distribution': {
                    str(i): len([c for c in scored_citations if c['llm_score'] == i])
                    for i in range(1, 6)
                },
                'average_score': sum(c['llm_score'] for c in scored_citations) / len(scored_citations) if scored_citations else 0
            }
        }
        
        # Output results
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo("\n" + "="*50)
            click.echo("RESULTS")
            click.echo("="*50)
            click.echo(f"Total citations found: {len(scored_citations)}")
            click.echo(f"Average LLM score: {results['summary']['average_score']:.2f}")
            click.echo("\nScore distribution:")
            for score, count in results['summary']['score_distribution'].items():
                click.echo(f"  Score {score}: {count} citations")
            
            click.echo("\nTop 5 citations:")
            for i, citation in enumerate(scored_citations[:5]):
                click.echo(f"\n{i+1}. Score: {citation['llm_score']}/5")
                click.echo(f"   Pages: {citation.get('start_page', '?')}-{citation.get('end_page', '?')}")
                click.echo(f"   Text: {citation['text'][:200]}...")
                click.echo(f"   LLM: {citation['llm_explanation']}")
    
    # Run the async workflow
    asyncio.run(run_workflow())


if __name__ == "__main__":
    main()
