# LexiconWeaver

A robust, human-in-the-loop framework for web novel translation with terminology consistency enforcement.

## Overview

LexiconWeaver addresses the critical problem of **Term Drift** in machine translation of web novels (Xianxia, LitRPG, Fantasy). Traditional translators optimize for sentence-level fluency, ignoring novel-wide consistency. This results in proper nouns being translated inconsistently across chapters (e.g., "Spirit Severing" → "Spirit Cutting" in Chapter 1, "Soul Split" in Chapter 2).

### The Solution

LexiconWeaver introduces a **Middleware Layer** that enforces terminology constraints *before* the AI generates text through a human-in-the-loop workflow:

1. **Scout**: The machine identifies potential terms using heuristics
2. **Annotate**: The human defines terms once
3. **Weave**: The machine generates text strictly adhering to those definitions

## Features

- **Intelligent Term Discovery**: Heuristic-based Scout engine identifies potential terms using frequency, capitalization, and structural patterns
- **Human-in-the-Loop Workflow**: Interactive TUI for efficient term annotation
- **Consistent Translation**: Dynamic glossary injection ensures terminology consistency
- **Translation Caching**: Avoid re-translating identical paragraphs
- **Dual Interface**: Both CLI for automation and TUI for interactive use
- **Robust Error Handling**: Graceful degradation and crash recovery

## Installation

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running with a language model

### Install LexiconWeaver

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Download Spacy Model (Optional)

For enhanced POS tagging:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Configure LexiconWeaver

Create a configuration file (or use the template):

```bash
lexiconweaver config init
lexiconweaver config path
```

Or manually create `~/.config/lexiconweaver/config.toml`:

```toml
[ollama]
url = "http://localhost:11434"
model = "llama2"
timeout = 300

[database]
path = ""  # Uses default location if empty

[scout]
min_confidence = 0.3
max_ngram_size = 4
```

### 2. Create a Project

```bash
lexiconweaver project create "My Novel"
```

### 3. Launch TUI

```bash
lexiconweaver tui chapter1.txt --project "My Novel"
```

### 4. Use CLI Commands

**Discover terms:**
```bash
lexiconweaver scout chapter1.txt --project "My Novel"
```

**Translate:**
```bash
lexiconweaver translate chapter1.txt --project "My Novel" --output translated.txt
```

## Usage

### TUI Interface

The TUI provides an interactive workspace:

- **Left Panel**: Chapter text with highlighting
  - Green: Confirmed terms (already in glossary)
  - Yellow: Candidate terms (suggested by Scout)
- **Right Panel**: Candidate queue sorted by confidence
- **Keybindings**:
  - `R`: Run Scout to discover terms
  - `Enter`: Edit/Confirm selected candidate
  - `Del`: Ignore selected candidate
  - `S`: Skip candidate
  - `Q`: Quit

### CLI Commands

```bash
# Config
lexiconweaver config path
lexiconweaver config init [--force]

# Project management
lexiconweaver project create <name>
lexiconweaver project list
lexiconweaver project select <name>
lexiconweaver project delete <name>

# Term discovery
lexiconweaver scout <file> [--project <name>] [--output <file>] [--min-confidence <0.0-1.0>]

# Translation
lexiconweaver translate <file> [--project <name>] [--output <file>]

# Launch TUI
lexiconweaver tui [<file>] [--project <name>]
```

## Architecture

LexiconWeaver consists of three core engines:

1. **Scout (Discovery Engine)**: Heuristic-based term discovery with confidence scoring
2. **Annotator (Interaction Engine)**: Textual-based TUI for term management
3. **Weaver (Generation Engine)**: LLM translation with dynamic glossary injection

### Data Flow

```
Raw Text → Scout → Candidate List → Annotator (User) → Glossary DB → Weaver → Translated Text
```

## Configuration

Configuration can be provided via:

1. **TOML file**: `~/.config/lexiconweaver/config.toml` (Linux/Mac) or `%APPDATA%/lexiconweaver/config.toml` (Windows)
2. **Environment variables**: Prefix with `LEXICONWEAVER_` (e.g., `LEXICONWEAVER_OLLAMA__URL=http://localhost:11434`)

**Config commands:**
- `lexiconweaver config path` — Show where config is read from
- `lexiconweaver config init` — Write a template config file (use `--force` to overwrite)

See `config/default.toml` for all available options.

## Development

### Setup Development Environment

```bash
git clone <repository>
cd WeaveCodex
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=lexiconweaver --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/lexiconweaver

# Formatting
black .
```

## Project Structure

```
WeaveCodex/
├── src/lexiconweaver/     # Main package
│   ├── database/          # Database models and management
│   ├── engines/           # Scout and Weaver engines
│   ├── tui/               # Textual TUI interface
│   ├── cli/               # CLI commands
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation
└── config/                # Configuration templates
```

## License

AGPLv3 License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
