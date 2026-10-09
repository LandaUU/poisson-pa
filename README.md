# Wang-Resnick Poisson Preferential Attachment Experiments

## EN

This repository contains an implementation of the Wang-Resnick model, numerical experiments for `N_k`, `|S_k|`, `|S_k| / N_k`, experiments at a random time `K*`, and model fitting for real SNAP temporal networks.

### Contents

- `main.py` - base implementation of graph evolution and message propagation.
- `scripts/run_nk_experiment.py`, `scripts/run_nk_grid.py` - checks of theoretical moments and the distribution of `N_k`.
- `scripts/run_pn_experiment.py` - checks of the local successful-transmission probability `p_n` for a fixed graph state.
- `scripts/run_sk_experiment.py`, `scripts/run_sk_grid.py` - simulations of `|S_k|` and relative coverage `|S_k| / N_k`.
- `scripts/run_sk_kstar_experiment.py`, `scripts/run_sk_kstar_grid.py` - simulations at a random time `K* ~ Poisson(nu T*)`.
- `model_fitting.py`, `scripts/run_collegemsg_fit.py`, `scripts/run_real_dataset_tail_comparison.py` - model fitting and diagnostics for real temporal networks.
- `notebooks/wang_resnick_visuals.ipynb` - Jupyter notebook for interactive figure generation.
- `scripts/download_snap_data.py` - downloader for the SNAP datasets used in the applied part of the project.
- `data/README.md` - input data list and source links.
- `scripts/render_supervisor_revision.py`, `scripts/empirical_intervals.py` - pointwise conditional Monte Carlo intervals from saved ensembles.
- `scripts/verify_gossip_rate_bound.py` - independent finite-state checks of a fixed-topology gossip completion bound.
- `scripts/run_empirical_smoke.py` - small current-model generation/rendering pipeline without manuscript archives.


Generated results are saved to `results/`. This directory is not versioned: artifacts can be reproduced by running the corresponding commands.

### Installation

The project uses Python and `uv`.

```bash
make sync
```

If `uv` is not installed, follow the official documentation: <https://docs.astral.sh/uv/>.

### Jupyter Notebook

For interactive figure generation in JupyterLab, run:

```bash
make notebook
```

The notebook `notebooks/wang_resnick_visuals.ipynb` uses the main repository code and saves generated images to `results/notebook/figures/`.

### SNAP Data Download

Raw SNAP files are not included in the repository. To download the datasets, run:

```bash
make snap-download
```

By default, this downloads:

- `data/CollegeMsg.txt.gz`
- `data/email-Eu-core-temporal.txt.gz`
- `data/sx-mathoverflow-a2q.txt.gz`

To download the files again, run:

```bash
uv run python -m scripts.download_snap_data --force
```

To download the additional departmental files `email-Eu-core-temporal-Dept1..4`, run:

```bash
uv run python -m scripts.download_snap_data --all
```

### Main Commands

Formatting and lint checks:

```bash
make check
```

Parameter-grid experiment for `N_k`:

```bash
make run-nk-grid
```

Experiment for the local probability `p_n`:

```bash
make run-pn
```

Main experiment for `|S_k|` and `|S_k| / N_k`:

```bash
make run-sk-grid
```

Experiment at the random time `K*`:

```bash
make run-sk-kstar-grid
```

Model fitting on CollegeMsg:

```bash
make snap-download
make run-collegemsg-fit
```

To run fitting for other SNAP datasets, use the `scripts.run_collegemsg_fit` module and pass `--data-path`, `--dataset-label`, and `--output-dir`.

Example for email-Eu-core temporal:

```bash
uv run python -m scripts.run_collegemsg_fit \
  --data-path data/email-Eu-core-temporal.txt.gz \
  --dataset-label "email-Eu-core temporal" \
  --output-dir results/email_eu_core_temporal_fit
```

Example for MathOverflow `a2q`:

```bash
uv run python -m scripts.run_collegemsg_fit \
  --data-path data/sx-mathoverflow-a2q.txt.gz \
  --dataset-label "MathOverflow a2q" \
  --output-dir results/mathoverflow_a2q_fit
```

### License

The code is distributed under the MIT license. See `LICENSE`.

## RU

Репозиторий содержит реализацию модели Ванга-Резника, численные эксперименты для величин `N_k`, `|S_k|`, `|S_k| / N_k`, эксперименты в случайный момент `K*`, а также fitting модели по реальным временным сетям SNAP.

### Содержание

- `main.py` - базовая реализация модели эволюции графа и распространения сообщения.
- `scripts/run_nk_experiment.py`, `scripts/run_nk_grid.py` - проверка теоретических моментов и распределения `N_k`.
- `scripts/run_pn_experiment.py` - проверка локальной вероятности успешной передачи `p_n` при фиксированном состоянии графа.
- `scripts/run_sk_experiment.py`, `scripts/run_sk_grid.py` - моделирование `|S_k|` и относительного охвата `|S_k| / N_k`.
- `scripts/run_sk_kstar_experiment.py`, `scripts/run_sk_kstar_grid.py` - моделирование в случайный момент `K* ~ Poisson(nu T*)`.
- `model_fitting.py`, `scripts/run_collegemsg_fit.py`, `scripts/run_real_dataset_tail_comparison.py` - fitting и диагностика на реальных временных сетях.
- `notebooks/wang_resnick_visuals.ipynb` - Jupyter notebook для интерактивной генерации графиков.
- `scripts/download_snap_data.py` - загрузка SNAP-датасетов, используемых в прикладной части.
- `data/README.md` - список входных данных и ссылки на источники.
- `scripts/render_supervisor_revision.py`, `scripts/empirical_intervals.py` — точечные условные Monte Carlo интервалы по сохранённым ансамблям.
- `scripts/verify_gossip_rate_bound.py` — независимая проверка границы завершения gossip на фиксированной топологии.
- `scripts/run_empirical_smoke.py` — малый запуск генерации ансамблей и построения интервалов без архива статьи.


Сгенерированные результаты сохраняются в `results/`. Эта директория не версионируется: артефакты воспроизводятся запуском соответствующих команд.

### Установка

Проект использует Python и `uv`.

```bash
make sync
```

Если `uv` не установлен, его можно поставить по инструкции из документации: <https://docs.astral.sh/uv/>.

### Jupyter Notebook

Для интерактивной генерации изображений через JupyterLab используйте:

```bash
make notebook
```

Notebook `notebooks/wang_resnick_visuals.ipynb` использует основной код репозитория и сохраняет сгенерированные изображения в `results/notebook/figures/`.

### Загрузка SNAP-Данных

Сырые SNAP-файлы не входят в репозиторий. Для загрузки датасетов выполните:

```bash
make snap-download
```

По умолчанию будут загружены:

- `data/CollegeMsg.txt.gz`
- `data/email-Eu-core-temporal.txt.gz`
- `data/sx-mathoverflow-a2q.txt.gz`

Для повторной загрузки используйте:

```bash
uv run python -m scripts.download_snap_data --force
```

Для загрузки дополнительных departmental-файлов `email-Eu-core-temporal-Dept1..4` используйте:

```bash
uv run python -m scripts.download_snap_data --all
```

### Основные Запуски

Проверка форматирования и lint:

```bash
make check
```

Эксперимент для `N_k` по сетке параметров:

```bash
make run-nk-grid
```

Эксперимент для локальной вероятности `p_n`:

```bash
make run-pn
```

Основной эксперимент для `|S_k|` и `|S_k| / N_k`:

```bash
make run-sk-grid
```

Эксперимент в случайный момент `K*`:

```bash
make run-sk-kstar-grid
```

Fitting модели на CollegeMsg:

```bash
make snap-download
make run-collegemsg-fit
```

Для запуска fitting на других SNAP-датасетах используйте модуль `scripts.run_collegemsg_fit`, передав `--data-path`, `--dataset-label` и `--output-dir`.

Пример для email-Eu-core temporal:

```bash
uv run python -m scripts.run_collegemsg_fit \
  --data-path data/email-Eu-core-temporal.txt.gz \
  --dataset-label "email-Eu-core temporal" \
  --output-dir results/email_eu_core_temporal_fit
```

Пример для MathOverflow `a2q`:

```bash
uv run python -m scripts.run_collegemsg_fit \
  --data-path data/sx-mathoverflow-a2q.txt.gz \
  --dataset-label "MathOverflow a2q" \
  --output-dir results/mathoverflow_a2q_fit
```

### License

Код распространяется под лицензией MIT. См. `LICENSE`.
