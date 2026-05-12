# article-xas-xncd

Small codebase and notebooks for **carbon K-edge XAS and XNCD** figures tied to a methox-type chiral system: **gas phase**, **water**, and **perfluorohexane** environments, using **CAM-B3LYP** as the main functional and **B3LYP** for supporting plots. Spectra are built from **WaveT**-style project folders (averaged linear and dichroic response) plus **PMM** stick data bundled under `data/spettro-*` and `data/spettri-singoli/`.

## What lives here

- **`notebooks/plots.ipynb`** — main article-style figures (energy axes, PCM vs PMM water panels, solvent comparisons, conversion factors for PMM intensities).
- **`notebooks/generate_spectra.ipynb`** — spectrum generation / exploration workflow.
- **`data/paths.json`** — absolute paths to WaveT templates on your machine (gas, PCM solvents, etc.) and pointers to local PMM bundles; **you must edit this after cloning**.
- **`src/article_xas_xncd/class_spectrum.py`** — `Spectrum` helper used by the notebooks to load projects and build averaged XAS/XNCD curves.

`results/` is for exports; large **local caches** (e.g. under `data/.spectrum_cache/`) are gitignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Use **Python ≥ 3.10** if you follow `pyproject.toml`; older interpreters may still run the notebooks if your stack allows it.

## Running

Open the notebooks with Jupyter (or VS Code / Cursor). There is a minimal CLI placeholder:

```bash
python -m article_xas_xncd.main
```

## Notes

- **`paths.json` is machine-specific.** Cloning on another host will not reproduce WaveT paths under `/data/...` or local reference spectra unless you update the JSON.
- Experimental and AMS reference paths in `paths.json` point to files outside this repo; add them or change keys if you do not need those overlays.

Public mirror: [github.com/Spoksonat/article-xas-xncd](https://github.com/Spoksonat/article-xas-xncd).
