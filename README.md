# PAN 2026 — RoBERTa fine-tuned

Main model submission for the PAN 2026 Voight-Kampff AI Detection task.
Fine-tuned RoBERTa-base classifier; weights hosted on HuggingFace at
`eljosey40/roberta-finetuned-pan26-voightkampff`.

Submission by Jose Alejandro Perez Dominguez
Master en Inteligencia Artificial — Universidad Europea de Valencia

## Usage

    python predict.py /path/to/dataset.jsonl /path/to/output_dir

Output: `predictions.jsonl` with `{"id": "...", "score": 0.XXXX}` per line.
Score > 0.5 = AI-generated, score < 0.5 = human-written.
