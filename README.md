# LLM from scratch

Proyecto de aprendizaje para construir, pieza a pieza, un modelo de lenguaje causal con PyTorch. La base es un decoder tipo GPT con embeddings posicionales aprendidos.

## Instalar

Requiere Python 3.10 o posterior. Desde la raíz del repositorio:

```powershell
python -m pip install -e ".[notebooks]"
```

## Entrenar y generar

Prepara un archivo UTF-8 de texto y ejecuta:

```powershell
python scripts/train.py --text-file data.txt --output checkpoints/gpt.pt
python scripts/generate.py --checkpoint checkpoints/gpt.pt --prompt "Érase una vez" --tokens 80
```

Los valores predeterminados son pequeños para poder experimentar en CPU. Ajusta `--block-size`, `--embedding-dim`, `--heads`, `--layers`, `--epochs` y `--batch-size` según el equipo y el corpus.

## Estructura

- `notebooks/`: recorrido didáctico de datos, attention, arquitectura, preentrenamiento y fine-tuning.
- `src/llm_from_scratch/`: tokenizer y dataset, attention causal, GPT, entrenamiento, generación, clasificación y LoRA.
- `scripts/`: puntos de entrada para entrenar, generar y ajustar el modelo.
- `tests/`: pruebas de forma y causalidad para attention y modelo.

El tokenizer incluido es sencillo y separa palabras y signos de puntuación; sirve para aprender y ejecutar el ejemplo, no pretende sustituir un tokenizer subword de producción. El notebook original de atención se conserva como `notebooks/03_attention.ipynb`. La imagen de referencia está en `docs/transformer.png`.
