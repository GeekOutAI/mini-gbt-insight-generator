Quickstart

	git clone https://github.com/GeekOutAI/mini-gbt-insight-generator.git
	cd mini-gbt-insight-generator

	python -m venv .venv
	. .venv\Scripts\Activate.ps1        # PowerShell
	pip install -r requirements.txt

	python -m streamlit run app.py

	Open http://localhost:8501
	and upload
	sample_data/sample_reviews.csv (provided)

Architecture

	flowchart LR
	A[CSV Upload] --> B[Preprocess]
	B --> C[Sentiment: DistilBERT]
	B --> D[Keyword Extraction: KeyBERT]
	B -->|optional| E[Topic Clustering: UMAP + HDBSCAN]
	C --> F[Metrics: Positive/Negative %]
	D --> F
	E --> F
	F --> G[Summarization: BART or OpenAI GPT]
	G --> H[Streamlit Dashboard + CSV Export]

Example Output

	| Metric         | Value |
	| -------------- | ----- |
	| Total Comments | 20    |
	| Positive %     | 40 %  |
	| Negative %     | 60 %  |

Executive Summary Example

	Positive highlights: Clean UI, fast dashboards, improved refresh speed.
	Negative highlights: Crashes on export, cluttered mobile UI, confusing billing page.
	
Tech Stack

	| Layer            | Tools                                  |
	| ---------------- | -------------------------------------- |
	| **Frontend/UI**  | Streamlit                              |
	| **ML/NLP**       | Hugging Face Transformers, KeyBERT     |
	| **Models**       | DistilBERT (sentiment), BART (summary) |
	| **Optional LLM** | OpenAI GPT                             |
	| **Data**         | CSV input                              |
	| **Exports**      | CSV download, summary text             |

Future Enhancements
	|Streamlit Cloud deployment with public demo
	|Multi-language sentiment models	
	|Interactive clustering visualization
	|GPT-generated “Pain-Points → Actions” insights

License
	MIT © 2025 Amy McCall (GeekOutAI)

Acknowledgments
	Open-source models from Hugging Face and community contributors.
	Special thanks to early testers for helping refine UX and summary prompts.

