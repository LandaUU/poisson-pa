# SNAP Data

## EN

Raw SNAP datasets are not stored in the GitHub repository. To download them, run:

```bash
make snap-download
```

The script `scripts/download_snap_data.py` downloads the files into this directory and checks the basic `SRC DST UNIXTS` format.

### Default Files

- `CollegeMsg.txt.gz` - private messages in an online social network at UC Irvine.
- `email-Eu-core-temporal.txt.gz` - emails between members of a European research institution.
- `sx-mathoverflow-a2q.txt.gz` - MathOverflow answers to questions.

### Sources

- CollegeMsg: <https://snap.stanford.edu/data/CollegeMsg.html>
- email-Eu-core temporal: <https://snap.stanford.edu/data/email-Eu-core-temporal.html>
- MathOverflow temporal: <https://snap.stanford.edu/data/sx-mathoverflow.html>

### Data Citation

CollegeMsg:

Panzarasa, P., Opsahl, T., and Carley, K. M. Patterns and Dynamics of Users' Behavior and Interaction: Network Analysis of an Online Community. Journal of the American Society for Information Science and Technology 60(5), 911-932, 2009.

email-Eu-core temporal and MathOverflow temporal:

Paranjape, A., Benson, A. R., and Leskovec, J. Motifs in Temporal Networks. Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, 601-610, 2017.

## RU

Сырые SNAP-датасеты не хранятся в GitHub-репозитории. Для загрузки используйте:

```bash
make snap-download
```

Скрипт `scripts/download_snap_data.py` скачивает файлы в эту директорию и проверяет базовый формат `SRC DST UNIXTS`.

### Файлы По Умолчанию

- `CollegeMsg.txt.gz` - private messages in an online social network at UC Irvine.
- `email-Eu-core-temporal.txt.gz` - emails between members of a European research institution.
- `sx-mathoverflow-a2q.txt.gz` - MathOverflow answers to questions.

### Источники

- CollegeMsg: <https://snap.stanford.edu/data/CollegeMsg.html>
- email-Eu-core temporal: <https://snap.stanford.edu/data/email-Eu-core-temporal.html>
- MathOverflow temporal: <https://snap.stanford.edu/data/sx-mathoverflow.html>

### Цитирование Данных

CollegeMsg:

Panzarasa, P., Opsahl, T., and Carley, K. M. Patterns and Dynamics of Users' Behavior and Interaction: Network Analysis of an Online Community. Journal of the American Society for Information Science and Technology 60(5), 911-932, 2009.

email-Eu-core temporal and MathOverflow temporal:

Paranjape, A., Benson, A. R., and Leskovec, J. Motifs in Temporal Networks. Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, 601-610, 2017.
