# article_xas_xncd

Proyecto base para analisis y desarrollo de scripts de espectroscopia (XAS/XNCD).

## Estructura

- `src/`: codigo fuente del paquete principal.
- `data/raw/`: datos originales sin procesar.
- `data/processed/`: datos transformados para analisis.
- `results/`: figuras, tablas y salidas finales.
- `notebooks/`: notebooks exploratorios.
- `scripts/`: utilidades ejecutables (pipeline, conversion, etc.).
- `tests/`: pruebas automaticas.
- `docs/`: notas tecnicas o documentacion adicional.

## Entorno virtual

Crear (ya creado en este repo):

```bash
python3 -m venv .venv
```

Activar:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Uso rapido

Ejecutar script base:

```bash
python -m article_xas_xncd.main
```

Ejecutar tests:

```bash
pytest -q
```

## Notas

- Guarda resultados reproducibles en `results/`.
- Evita modificar datos originales en `data/raw/`.
