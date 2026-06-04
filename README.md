# Wang-Resnick Poisson Preferential Attachment Experiments

Репозиторий содержит реализацию модели Ванга-Резника, численные эксперименты для величин `N_k`, `|S_k|`, `|S_k| / N_k`, эксперименты в случайный момент `K*`, а также fitting модели по реальным временным сетям SNAP.

## Содержание

- `main.py` - базовая реализация модели эволюции графа и распространения сообщения.
- `scripts/run_nk_experiment.py`, `scripts/run_nk_grid.py` - проверка теоретических моментов и распределения `N_k`.
- `scripts/run_pn_experiment.py` - проверка локальной вероятности успешной передачи `p_n` при фиксированном состоянии графа.
- `scripts/run_sk_experiment.py`, `scripts/run_sk_grid.py` - моделирование `|S_k|` и относительного охвата `|S_k| / N_k`.
- `scripts/run_sk_kstar_experiment.py`, `scripts/run_sk_kstar_grid.py` - моделирование в случайный момент `K* ~ Poisson(nu T*)`.
- `model_fitting.py`, `scripts/run_collegemsg_fit.py`, `scripts/run_real_dataset_tail_comparison.py` - fitting и диагностика на реальных временных сетях.
- `notebooks/wang_resnick_visuals.ipynb` - Jupyter notebook для интерактивной генерации графиков.
- `scripts/download_snap_data.py` - загрузка SNAP-датасетов, используемых в прикладной части.
- `data/README.md` - список входных данных и ссылки на источники.

Сгенерированные результаты сохраняются в `results/`. Эта директория не версионируется: артефакты воспроизводятся запуском соответствующих команд.

## Установка

Проект использует Python и `uv`.

```bash
make sync
```

Если `uv` не установлен, его можно поставить по инструкции из документации: <https://docs.astral.sh/uv/>.

## Jupyter Notebook

Для интерактивной генерации изображений через JupyterLab используйте:

```bash
make notebook
```

Notebook `notebooks/wang_resnick_visuals.ipynb` использует основной код репозитория и сохраняет сгенерированные изображения в `results/notebook/figures/`.

## Загрузка SNAP-Данных

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

## Основные Запуски

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

## License

Код распространяется под лицензией MIT. См. `LICENSE`.
