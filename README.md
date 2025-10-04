# Citation Workflow CLI

A command-line tool for extracting text from PDFs, finding citations related to specific events, and scoring them using AI models.

## Features

- **PDF Text Extraction**: Uses BookWyrm to extract structured text from PDF documents
- **Character-to-Page Mapping**: Maintains mapping from character positions in concatenated text to original page numbers
- **Phrasal Processing**: Uses BookWyrm's phrasal model to break text into meaningful chunks
- **Citation Finding**: Uses BookWyrm's citation model to find evidence related to query events
- **LLM Scoring**: Uses OpenAI's GPT-3.5-turbo to score citations on a 1-5 scale (1=unrelated, 5=comprehensive)

## Installation

```bash
uv sync
```

## Configuration

Set up your API keys as environment variables:

```bash
export BOOKWYRM_API_KEY="your-bookwyrm-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Alternatively, create a `.env` file in the project root:

```
BOOKWYRM_API_KEY=your-bookwyrm-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Usage

```bash
uv run citation-workflow PDF_PATH QUERY_DOCUMENT [OPTIONS]
```

### Arguments

- `PDF_PATH`: Path to the PDF file to analyze
- `QUERY_DOCUMENT`: Path to text file describing the event to search for

### Options

- `--output, -o`: Output JSON file path (optional, prints to console if not specified)
- `--bookwyrm-api-key`: BookWyrm API key (overrides environment variable)
- `--openai-api-key`: OpenAI API key (overrides environment variable)
- `--help`: Show help message

### Example

```bash
# Analyze a document for climate change impacts
uv run citation-workflow research_paper.pdf query_climate.txt --output results.json
```

### Query Document Format

Create a text file describing the event or topic you want to find citations for:

```txt
Describe any mentions of climate change impacts on agriculture, including changes in crop yields, farming practices, or agricultural adaptation strategies.
```

## Output

The tool outputs either to console or JSON file with:

- **Query**: The search query used
- **Citations**: Found citations with:
  - Original text and reasoning from BookWyrm
  - LLM score (1-5) and explanation
  - Page numbers where the citation appears
  - BookWyrm quality score
- **Summary**: Score distribution and average score

### Example Output

```json
{
  "query": "climate change impacts on agriculture",
  "pdf_path": "research_paper.pdf",
  "total_citations": 12,
  "citations": [
    {
      "text": "Climate change has significantly reduced corn yields in the Midwest...",
      "reasoning": "This directly addresses agricultural impacts of climate change",
      "quality": 4,
      "llm_score": 5,
      "llm_explanation": "5 - Comprehensive account that fully addresses the query",
      "start_page": 15,
      "end_page": 15
    }
  ],
  "summary": {
    "score_distribution": {
      "1": 0,
      "2": 1,
      "3": 3,
      "4": 5,
      "5": 3
    },
    "average_score": 3.8
  }
}
```

## Workflow Steps

1. **PDF Extraction**: Extracts text from PDF while preserving page structure
1. **Text Processing**: Uses BookWyrm's phrasal model to create meaningful text chunks
1. **Citation Search**: Finds relevant citations using BookWyrm's citation model
1. **LLM Scoring**: Each citation is scored by GPT-3.5-turbo on relevance (1-5 scale)
1. **Results**: Citations are sorted by LLM score and include page references

## API Requirements

- **BookWyrm API**: For PDF extraction, text processing, and citation finding
- **OpenAI API**: For scoring citations with GPT-3.5-turbo

## Development

The main CLI logic is in `src/citation_workflow/main.py`. Key components:

- `CharacterToPageMapper`: Maps character positions to page numbers
- `extract_pdf_text()`: Extracts and maps PDF text
- `process_text_to_chunks()`: Creates text chunks using phrasal model
- `find_citations()`: Finds relevant citations
- `score_citations_with_llm()`: Scores citations with OpenAI
