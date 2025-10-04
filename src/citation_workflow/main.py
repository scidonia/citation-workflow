import asyncio
import json
import os
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


async def extract_pdf_text(pdf_path: str, api_key: Optional[str] = None) -> Tuple[str, CharacterToPageMapper]:
    """Extract text from PDF and create character-to-page mapping."""
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
        return raw_text, mapper


async def process_text_to_chunks(text: str, api_key: Optional[str] = None) -> List[TextSpan]:
    """Process raw text into chunks using BookWyrm's phrasal model."""
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
    return chunks


async def find_citations(chunks: List[TextSpan], query: str, api_key: Optional[str] = None) -> List[dict]:
    """Find citations related to the query event."""
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
    
    return citations


def score_citations_with_llm(citations: List[dict], query: str, openai_api_key: Optional[str] = None) -> List[dict]:
    """Score citations using OpenAI LLM (1-5 scale)."""
    click.echo("Scoring citations with LLM...")
    
    client = OpenAI(api_key=openai_api_key)
    
    scored_citations = []
    
    for i, citation in enumerate(citations):
        click.echo(f"Scoring citation {i+1}/{len(citations)}")
        
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
            scored_citations.append(citation_with_score)
            
        except Exception as e:
            click.echo(f"Error scoring citation: {e}")
            citation_with_score = citation.copy()
            citation_with_score['llm_score'] = 3  # Default score
            citation_with_score['llm_explanation'] = f"Error during scoring: {e}"
            scored_citations.append(citation_with_score)
    
    return scored_citations


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.argument('query_document', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file path')
@click.option('--bookwyrm-api-key', envvar='BOOKWYRM_API_KEY', help='BookWyrm API key')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def main(pdf_path: str, query_document: str, output: Optional[str], 
         bookwyrm_api_key: Optional[str], openai_api_key: Optional[str]):
    """
    Citation workflow CLI.
    
    Extracts text from PDF, processes it with BookWyrm's phrasal model,
    finds citations related to events described in the query document,
    and scores them with an LLM.
    
    PDF_PATH: Path to the PDF file to analyze
    QUERY_DOCUMENT: Path to text file describing the event to search for
    """
    
    async def run_workflow():
        # Read query document
        with open(query_document, 'r', encoding='utf-8') as f:
            query = f.read().strip()
        
        click.echo(f"Query: {query}")
        
        # Step 1: Extract PDF text with page mapping
        raw_text, page_mapper = await extract_pdf_text(pdf_path, bookwyrm_api_key)
        
        # Step 2: Process text into chunks
        chunks = await process_text_to_chunks(raw_text, bookwyrm_api_key)
        
        # Step 3: Find citations
        citations = await find_citations(chunks, query, bookwyrm_api_key)
        
        # Step 4: Score citations with LLM
        scored_citations = score_citations_with_llm(citations, query, openai_api_key)
        
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
