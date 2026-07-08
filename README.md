# Личный сайт Александры Скворцовой

Базовый Flask-проект для сайта-визитки и проектного журнала. Сайт представляет Александру Скворцову как специалиста на стыке закупок, логистики, бизнес-процессов, данных и ИИ.

## Структура

```text
.
├── app.py
├── static/
│   └── css/
│       └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── projects.html
    ├── project_detail.html
    ├── about.html
    ├── contacts.html
    └── partials/
        └── project_card.html
```

## Запуск

1. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Запустите сайт:

```bash
python app.py
```

4. Откройте в браузере:

```text
http://127.0.0.1:5000
```

## Где менять проекты

Данные проектов пока хранятся в списке `PROJECTS` в файле `app.py`. На следующем этапе их можно вынести в JSON-файл или подключить базу данных.

## Индексация базы знаний

Markdown-файлы для базы знаний лежат в папке `knowledge_base/`. Чтобы пересобрать ChromaDB-индекс для RAG-ядра сайта, заполните `.env` по примеру `.env.example` и запустите:

```bash
python scripts/ingest_site_knowledge.py
```
