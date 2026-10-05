# extract-rust

**English** | [Español](#español)

Source-grounded information extraction library and CLI in Rust.

`extract-rust` extracts structured entities from unstructured text using Large Language Models (LLMs) and grounds every extraction back to its exact character and token offsets in the source text.

> **Notice**: This project is an independent Rust port and implementation of the extraction and alignment algorithm inspired by Google LangExtract (Apache-2.0). It is **not** an official Google project or clone.

---

## Status

- [x] **Tokenizer**: Regex and Unicode script/category tokenizers, sentence boundary detector (`find_sentence_range`).
- [x] **Chunking**: Sentence-aware text chunking with coreference context window overlap.
- [x] **Aligner**: DP exact substring alignment and LCS fuzzy alignment with density gates.
- [x] **Pipeline**: High-level `extract()` with multi-pass extraction and prompt alignment validation.
- [x] **HTTP Providers**: Ollama (local inference), Google Gemini REST, OpenAI Chat Completions.
- [x] **CLI**: `extract-cli` binary with JSON/YAML output and parallel batch workers.
- [x] **Visualization**: self-contained HTML player (`visualize()`, CLI `--html`).
- [x] **JSONL IO**: save/load annotated documents.
- [ ] **Missing**: Vertex AI batch processing.

---

## CLI Usage

Run the CLI using `cargo run -p extract-cli --`.

### 1. Ollama (Local)

Make sure Ollama is running locally (`ollama run gemma2:2b`):

```bash
cargo run -p extract-cli -- \
  -m gemma2:2b \
  -p "Extract people and their roles" \
  -e examples/people.json \
  --file examples/note.txt
```

### 2. Gemini

Set your Gemini API key (or pass `--api-key`):

```bash
export GEMINI_API_KEY="your-api-key"

cargo run -p extract-cli -- \
  -m gemini-2.5-flash \
  -p "Extract people and their roles" \
  -e examples/people.json \
  --file examples/note.txt
```

### 3. OpenAI

Set your OpenAI API key (or pass `--api-key`):

```bash
export OPENAI_API_KEY="your-api-key"

cargo run -p extract-cli -- \
  -m gpt-4o-mini \
  -p "Extract people and their roles" \
  -e examples/people.json \
  --file examples/note.txt
```

### Additional Flags

- Direct text: `--text "Ada Lovelace documented the Analytical Engine."`
- Save to file: `--output results.json`
- HTML review: `--html review.html`
- Output format: `--format yaml` (default: `json`)
- Custom Ollama / OpenAI URL: `--base-url http://localhost:11434`

---

## `examples.json` Format

Few-shot examples define the target schema. The file is a JSON array of `ExampleData` objects where each entity's text must appear verbatim in the sample text:

```json
[
  {
    "text": "Alan Turing was an English mathematician and computer scientist who worked at Bletchley Park.",
    "extractions": [
      {
        "extraction_class": "person",
        "extraction_text": "Alan Turing",
        "attributes": {
          "role": "mathematician and computer scientist",
          "workplace": "Bletchley Park"
        }
      }
    ]
  }
]
```

Fields:
- `text`: Example source text.
- `extractions`: List of target entities.
  - `extraction_class`: Entity label (e.g. `person`, `condition`, `organization`).
  - `extraction_text`: Exact verbatim span from `text`.
  - `attributes`: Optional key-value metadata map associated with the extraction.

---

## Crate API Sketch

Add `extract-core` and `extract-providers` to your `Cargo.toml`:

```rust
use extract_core::extract::{extract, ExtractInput, ExtractOptions, ExtractResult};
use extract_core::types::{ExampleData, Extraction};
use extract_providers::{create_model, ModelOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ExampleData::new(
            "Alan Turing worked at Bletchley Park.",
            vec![Extraction::new("person", "Alan Turing")],
        ),
    ];

    let mut model = create_model("gemma2:2b", ModelOptions::default())?;

    let result = extract(
        ExtractInput::Text("Ada Lovelace wrote the first algorithm.".into()),
        "Extract people mentioned in the source text",
        &examples,
        model.as_mut(),
        ExtractOptions::default(),
    )?;

    if let ExtractResult::Document(doc) = result {
        for ext in doc.extractions.unwrap_or_default() {
            println!(
                "Class: {} | Text: {} | Offsets: {:?}",
                ext.extraction_class,
                ext.extraction_text,
                ext.char_interval
            );
        }
    }

    Ok(())
}
```

---

## Español

Librería y CLI en Rust para extracción de información con anclaje al texto fuente (*source grounding*).

`extract-rust` extrae entidades estructuradas desde texto libre mediante modelos de lenguaje (LLMs) y asocia cada extracción a sus *offsets* exactos de caracteres y tokens en el texto original.

> **Aviso**: Esta es una implementación y port independiente en Rust del algoritmo de extracción y alineación inspirado en Google LangExtract (Apache-2.0). **No** es un clon ni un proyecto oficial de Google.

### Estado del Proyecto

- [x] **Tokenizer**: Tokenizadores Regex y Unicode por scripts/categorías, detección de oraciones (`find_sentence_range`).
- [x] **Chunking**: División de texto orientada a oraciones con ventana de contexto histórico.
- [x] **Aligner**: Alineación exacta por programación dinámica (DP) y difusa por subsecuencia común más larga (LCS).
- [x] **Pipeline**: Pipeline de alto nivel `extract()` con múltiples pasadas y validación previa de ejemplos.
- [x] **Proveedores HTTP**: Ollama (local), Google Gemini REST y OpenAI Chat Completions.
- [x] **CLI**: Herramienta de línea de comandos `extract-cli` con salida JSON/YAML y trabajadores paralelos.
- [x] **Visualización**: HTML autocontenido (`visualize()`, CLI `--html`).
- [x] **JSONL**: guardar y cargar documentos anotados.
- [ ] **Pendiente**: Procesamiento por lotes en Vertex AI (Vertex batch).

### Uso del CLI

Ejecuta el CLI con `cargo run -p extract-cli --`.

#### Ollama (Local)

```bash
cargo run -p extract-cli -- \
  -m gemma2:2b \
  -p "Extrae personas y sus roles" \
  -e examples/people.json \
  --file examples/note.txt
```

#### Gemini

```bash
export GEMINI_API_KEY="tu-api-key"

cargo run -p extract-cli -- \
  -m gemini-2.5-flash \
  -p "Extrae personas y sus roles" \
  -e examples/people.json \
  --file examples/note.txt
```

#### OpenAI

```bash
export OPENAI_API_KEY="tu-api-key"

cargo run -p extract-cli -- \
  -m gpt-4o-mini \
  -p "Extrae personas y sus roles" \
  -e examples/people.json \
  --file examples/note.txt
```

### Formato de `examples.json`

Define el esquema mediante ejemplos *few-shot*. El archivo contiene un arreglo JSON donde cada texto de extracción debe coincidir exactamente con el texto de ejemplo:

```json
[
  {
    "text": "Alan Turing fue un matemático británico que trabajó en Bletchley Park.",
    "extractions": [
      {
        "extraction_class": "persona",
        "extraction_text": "Alan Turing",
        "attributes": {
          "rol": "matemático",
          "lugar": "Bletchley Park"
        }
      }
    ]
  }
]
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
