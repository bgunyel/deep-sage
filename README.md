# Deep Sage

An AI-powered research tool that automatically generates comprehensive reports on any topic by intelligently searching the web and synthesizing information using large language models.

## Overview

Deep Sage is a sophisticated research automation system that leverages multiple AI components to create detailed, well-structured reports. It combines web search capabilities with advanced language models to gather, analyze, and synthesize information on any given topic.

## Features

- **Intelligent Planning**: Automatically creates structured research plans with targeted sections
- **Web Research**: Performs comprehensive web searches using Tavily API for up-to-date information
- **Content Generation**: Uses multiple LLM providers (Groq, OpenAI, Anthropic) for content creation
- **Async Processing**: Concurrent processing of multiple sections for improved performance
- **Source Management**: Automatic deduplication and citation of sources
- **Multiple Formats**: Generates both Markdown and PDF outputs
- **Token Tracking**: Comprehensive cost tracking across all LLM operations

## Architecture

The system follows a modular pipeline architecture with four main components:

1. **Planner**: Creates research sections and determines which require web research
2. **Sections Writer**: Generates content for research-based sections using web search
3. **Final Writer**: Creates introduction/conclusion sections and report title
4. **Finalizer**: Merges sources and assembles the final report

## Installation

### Prerequisites

- Python 3.13 or higher
- UV package manager (recommended) or pip

### Using UV (Recommended)

```bash
git clone https://github.com/bgunyel/deep-sage.git
cd deep-sage
uv sync
```

### Using pip

```bash
git clone https://github.com/bgunyel/deep-sage.git
cd deep-sage
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key  # Optional for tracing
LANGSMITH_TRACING=true  # Optional for tracing
```

## Usage

### Basic Usage

```python
from deep_sage import Researcher

# Configure LLM settings
llm_config = {
    'language_model': {
        'model': 'llama-3.3-70b-versatile',
        'model_provider': 'groq',
        'api_key': 'your_groq_api_key',
        'model_args': {
            'temperature': 0,
            'max_retries': 5,
            'max_tokens': 32768,
        }
    },
    'reasoning_model': {
        'model': 'deepseek-r1-distill-llama-70b',
        'model_provider': 'groq',
        'api_key': 'your_groq_api_key',
        'model_args': {
            'temperature': 0,
            'max_retries': 5,
            'max_tokens': 32768,
        }
    }
}

# Create researcher instance
researcher = Researcher(
    llm_config=llm_config,
    web_search_api_key='your_tavily_api_key'
)

# Generate report
result = researcher.run(
    topic="Impact of artificial intelligence on healthcare",
    config={
        "configurable": {
            "max_iterations": 3,
            "max_results_per_query": 5,
            "number_of_queries": 3,
            "search_category": "general"
        }
    }
)

print(result['content'])  # Markdown report
print(f"Report title: {result['report_title']}")
```

### Development Script

For quick testing, use the included development script:

```bash
python src/main_dev.py
```

This will generate a report and save both Markdown and PDF versions to the `out/` directory with timestamped filenames.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iterations` | Maximum research iterations | 3 |
| `max_results_per_query` | Results per search query | 5 |
| `max_tokens_per_source` | Token limit per source | 5000 |
| `number_of_queries` | Search queries to generate | 3 |
| `search_category` | Tavily search category | "general" |
| `strip_thinking_tokens` | Remove reasoning tokens | true |

## Supported LLM Providers

- **Groq**: Fast inference with Llama and other models
- **OpenAI**: GPT-4 and other OpenAI models
- **Anthropic**: Claude models

## Output Structure

Generated reports include:

- **Title**: AI-generated report title
- **Introduction**: Overview and context
- **Research Sections**: Data-driven content with sources
- **Conclusion**: Synthesis and key takeaways
- **Citations**: Automatically formatted numbered source list with clickable links

## Dependencies

- `langchain`: LLM framework and integrations
- `langgraph`: Workflow orchestration
- `tavily-python`: Web search API
- `pydantic`: Data validation and settings
- `md2pdf`: PDF generation
- `ai-common`: Shared utilities (custom package)
- `summary-writer`: Content generation (custom package)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Bertan Günyel**
- Email: bertan.gunyel@gmail.com
- GitHub: [@bgunyel](https://github.com/bgunyel)

## Acknowledgments

- Built with LangChain and LangGraph frameworks
- Uses Tavily for web search capabilities
- Supports multiple LLM providers for flexibility