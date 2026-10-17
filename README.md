# Выбор между TabPFN-2.5 и градиентным бустингом в кредитном скоринге

Статья для сборника НИТУ «МИСиС» (курс «Научное письмо и публикационная деятельность»).
Оформление: LaTeX-шаблон [sadekovlar/misis-article-latex](https://github.com/sadekovlar/misis-article-latex) (`misisarticle.cls`, LPPL 1.3c, без изменений).

## Структура

| Путь | Назначение |
| --- | --- |
| `article.tex` | Текст статьи |
| `misisarticle.cls` | Класс оформления (скопирован из шаблона, не менять) |
| `figures/` | Рисунки для статьи |
| `src/` | Код экспериментов (агрегация, подвыборки, модели, менеджер) |
| `notebooks/` | Ноутбуки с анализом |
| `data/` | Данные; в репозиторий не коммитятся (см. `.gitignore`) |

## Сборка статьи

Компилятор: XeLaTeX или tectonic.

```
make article        # PDF: build/article.pdf
make dist           # архив для Overleaf: dist/article.zip
```

В Overleaf: New Project, Upload Project, затем Menu, Compiler, XeLaTeX; главный файл `article.tex`.

## Данные

Собственные обезличенные данные о кредитных клиентах/заявках, полученные от банка и самостоятельно подготовленные для исследования.

## Подготовка данных

1. Скачайте `train_data_*.pq` и `train_target.csv` и положите в `data/raw/`.
2. Агрегация до одной строки на заявку (около 6 с на файл):
   ```
   pip install -r requirements.txt
   python src/aggregate.py --input data/raw --target data/raw/train_target.csv --out data/agg.parquet
   ```
   Результат: `data/agg.parquet` и `data/agg_meta.json` (список категориальных признаков).
3. Разбиение по времени и подвыборки:
   ```
   python src/make_splits.py --agg data/agg.parquet --out data/splits
   ```
   Валидация и тест ограничены 200 000 объектов (параметр `--eval-size`, `0` отключает ограничение): вывод TabPFN на миллионах строк очень медленный.
   Результат: `val.parquet`, `test.parquet`, папка `subsamples/` (размеры, доля дефолтов, повторы), `feature_sets.json` (вложенные наборы признаков), `manifest.csv`.

Особенности данных: числовые признаки `pre_*` бинаризованы со случайной нумерацией интервалов, поэтому среднее и максимум по ним не считаются, а используются первое и последнее значение и число различных значений. Столбцы `enc_paym_11`, `enc_paym_20`, `enc_paym_24` закодированы со сдвигом +1, скрипт это исправляет.

## Лицензия

Класс `misisarticle.cls` распространяется под LPPL 1.3c (см. `LICENSE`).
