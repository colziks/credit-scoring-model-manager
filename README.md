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

## Лицензия

Класс `misisarticle.cls` распространяется под LPPL 1.3c (см. `LICENSE`).
