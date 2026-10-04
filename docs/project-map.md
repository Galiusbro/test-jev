# Карта проекта: файл → за что отвечает

В репозитории два независимых проекта:

- **jev-agent** — сама система (ИИ-разработчик с правилами и проверками): `src/`, `tests/`, корневые конфиги.
- **demo-api** — подопытное приложение, над которым работает агент: `demo-api/`. Свой `pyproject.toml`, `uv.lock` и `.venv`; jev-agent его не импортирует.

Путь одного прогона:

```
ticket → prepare_workspace → triage (Jev) → plan → policy_check → implement
       → validate → prove_tests → review → report → [PR]
                ↘ при неудаче: diagnose (Jev) → repair (≤3, затем модель strong)
```

---

## 1. Точка входа и запуск

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/cli.py](../src/jev_agent/cli.py) | Все команды `jev-agent`: `doctor`, `run`, `init`, `mcp`, `eval`, `eval-report`, `bench`, `models`, `ask`. Только разбор аргументов и вывод в консоль, логика лежит в модулях ниже. |
| [src/jev_agent/runner.py](../src/jev_agent/runner.py) | Один прогон, общий для `run` и `eval`: собирает `RunConfig`, включает трассировку, стримит граф, превращает сбой LLM в статус `error` с отчётом. Здесь же режимы согласия (`ask` / `all` / `none`) и включение Jev. |
| [src/jev_agent/config.py](../src/jev_agent/config.py) | Все настройки из `.env`: ключи NVIDIA, TypeSafe и LangSmith; цепочки моделей по уровням (fast / strong / coder); режимы рассуждения; таймауты; лимит repair. |
| [src/jev_agent/tickets.py](../src/jev_agent/tickets.py) | Модель задачи `Ticket` и чтение задачи из Markdown (первый `# заголовок` — название, остальное — описание). |

## 2. Процесс (оркестрация)

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/graph.py](../src/jev_agent/graph.py) | **Сердце системы.** Граф LangGraph: узлы (`prepare_workspace`, `triage`, `plan`, `policy_check`, `implement`, `validate`, `prove_tests`, `review`, `diagnose`, `repair`, `report`), маршрутизация между ними (`after_*`), лестница эскалации repair, итоговый статус (`final_status`), метрики и запись `report.json` / `changes.diff`. |
| [src/jev_agent/agents.py](../src/jev_agent/agents.py) | Что делает каждый LLM-агент: промпты planner, implementer, repair и reviewer; схема `Plan` (не может быть пустой); структурированный вывод с исправляющей повторной попыткой (`structured`); схема `Review` и привязка замечаний к diff (`grounded`). |
| [src/jev_agent/harness.py](../src/jev_agent/harness.py) | Как агент крутится: LangChain `create_agent` + middleware. Лимит шагов, сжатие контекста (последнее прочтение каждого файла сохраняется), возврат невалидных вызовов инструментов модели, счётчик шагов, Jev-router модели, Jev-проверка каждой записи. |
| [src/jev_agent/project.py](../src/jev_agent/project.py) | Читает AGENTS.md целевого репозитория: разделы `## Commands` (проверки) и `## Autofix` (форматирование). |

## 3. Безопасность и границы

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/policy.py](../src/jev_agent/policy.py) | **Правила из AGENTS.md в коде.** Разбор разделов Allowed / Approval required / Forbidden в объекты `Rule`; проверка каждого чтения и записи; проверка плана; защита существующих тестов (сравнение AST с базовым коммитом); согласие человека; журнал решений. |
| [src/jev_agent/tools.py](../src/jev_agent/tools.py) | Возможности агента вместо shell: `list_files`, `read_file`, `search`, `write_file`, `replace_in_file`, `run_check`. Нормализация путей перед проверкой правил, защита синтаксиса Python, `run_check` только для объявленных команд. |
| [src/jev_agent/workspace.py](../src/jev_agent/workspace.py) | Изолированная копия репозитория (`runs/<id>/repo`) с базовым git-коммитом; запрет выхода за её пределы; diff, базовая версия файла, запуск команд с таймаутом и чистым окружением. |
| [src/jev_agent/proof.py](../src/jev_agent/proof.py) | Доказательство регрессии: временно возвращает исходный код и запускает новые тесты — они обязаны упасть. |

## 4. Решения Jev

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/decisions/fabric.py](../src/jev_agent/decisions/fabric.py) | **Все решения Jev на переходах графа**, у каждого порог уверенности и запасной вариант: `triage`, `assess_plan` (риск + сложность), `write_gate`, `route_model`, `diagnose`, `verify_findings`. Журнал решений. |
| [src/jev_agent/decisions/client.py](../src/jev_agent/decisions/client.py) | Связь с Jev через официальный `TypeSafeClassifier` (`langchain-typesafe`): преобразование типов, повторы при 429/5xx, имя запуска для LangSmith. `FakeJevClient` для тестов. |
| [src/jev_agent/decisions/types.py](../src/jev_agent/decisions/types.py) | Типы вопросов (yes/no, choice, score) и ответов; confidence для yes/no вычисляется как \|2p−1\|. |
| [src/jev_agent/decisions/\_\_init\_\_.py](../src/jev_agent/decisions/__init__.py) | Публичный экспорт пакета. |

## 5. Модели (LLM)

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/llm/resilient.py](../src/jev_agent/llm/resilient.py) | `ResilientChatModel` — модель LangChain поверх NVIDIA API: потоковый режим, таймаут первого токена (зависит от размера запроса), цепочка запасных моделей, второй проход, временное понижение упавшей модели, починка вызовов инструментов (написанных текстом, с битыми скобками, с неверным регистром), журнал вызовов с токенами. |
| [src/jev_agent/llm/reasoning.py](../src/jev_agent/llm/reasoning.py) | Как включить или выключить рассуждение у каждого семейства моделей (`enable_thinking` у Nemotron и Laguna, `reasoning_effort` у gpt-oss). |
| [src/jev_agent/llm/\_\_init\_\_.py](../src/jev_agent/llm/__init__.py) | `chat_model(tier, settings)` — единственный способ получить модель; `list_models`. |
| [src/jev_agent/bench.py](../src/jev_agent/bench.py) | Замер моделей: задержка и правильность вызова инструментов (`jev-agent bench`); по нему выбраны цепочки. |

## 6. Интеграции

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/pr.py](../src/jev_agent/pr.py) | PR на GitHub: временный `git worktree`, новая ветка, `git apply` в подкаталог, commit, push, `gh pr create`; описание PR из отчёта. |
| [src/jev_agent/mcp_server.py](../src/jev_agent/mcp_server.py) | MCP-сервер (`jev-agent mcp`): те же инструменты с теми же правилами для любого MCP-клиента (Claude Code, Cursor), плюс `project_rules` и `triage_ticket`. |
| [src/jev_agent/observability.py](../src/jev_agent/observability.py) | Трассировка в LangSmith: экспорт ключа из `.env`, ссылка на трассу, ожидание отправки. |
| [src/jev_agent/scaffold.py](../src/jev_agent/scaffold.py) | `jev-agent init`: определяет стек и создаёт AGENTS.md (в формате для policy engine), CLAUDE.md и `.mcp.json`; ничего не перезаписывает. |

## 7. Оценка

| Файл | За что отвечает |
|---|---|
| [src/jev_agent/evals.py](../src/jev_agent/evals.py) | Серия прогонов (задачи × режимы × повторы), запись в JSONL с продолжением после обрыва, оценка (правильно / вредно / сбой провайдера / запрещённое в diff), сводные таблицы. |
| [evals/cases.json](../evals/cases.json) | Набор задач для оценки: ожидаемые статусы, вредные статусы, запрещённые шаблоны в diff, режим согласия. |
| [evals/results/latest.jsonl](../evals/results/latest.jsonl) | Итоговая серия «с Jev против без Jev» (основа отчёта). |
| [evals/results/2026-10-04-old-code.jsonl](../evals/results/2026-10-04-old-code.jsonl) | Серия на старом коде до исправлений (для сравнения). |
| [evals/results/security.jsonl](../evals/results/security.jsonl), [security-007-rerun.jsonl](../evals/results/security-007-rerun.jsonl) | Проверка безопасности: задачи 001 и 007 после чек-листа reviewer. |
| [docs/evaluation-report.md](evaluation-report.md) | Отчёт об оценке: цифры, разбор случаев, выводы, ограничения. |

## 8. Задачи для агента

| Файл | Что проверяет |
|---|---|
| [tickets/001-login-rate-limit.md](../tickets/001-login-rate-limit.md) | Фича: ограничение неудачных входов (429), нужно согласие на изменение API. |
| [tickets/002-make-login-better.md](../tickets/002-make-login-better.md) | Расплывчатая задача → должна уйти человеку. |
| [tickets/003-deploy-to-production.md](../tickets/003-deploy-to-production.md) | Вне рамок + секреты → стоп. |
| [tickets/004-trim-user-names.md](../tickets/004-trim-user-names.md) | Простой багфикс (из него [PR #1](https://github.com/Galiusbro/test-jev/pull/1)). |
| [tickets/005-skip-flaky-test.md](../tickets/005-skip-flaky-test.md) | Ловушка: просьба отключить тест. |
| [tickets/006-add-user-roles.md](../tickets/006-add-user-roles.md) | Миграция БД + API → отказ без согласия. |
| [tickets/007-rate-limit-behind-proxy.md](../tickets/007-rate-limit-behind-proxy.md) | Безопасность: реальный IP за прокси без доверия к заголовкам (из него [PR #2](https://github.com/Galiusbro/test-jev/pull/2)). |

## 9. demo-api (подопытное приложение)

| Файл | За что отвечает |
|---|---|
| [demo-api/AGENTS.md](../demo-api/AGENTS.md) | **Правила для агента:** команды, autofix, конвенции, Allowed / Approval required / Forbidden, definition of done. Вход для `project.py` и `policy.py`. |
| [demo-api/src/demo_api/main.py](../demo-api/src/demo_api/main.py) | Фабрика приложения FastAPI, подключение сервисов через `app.state`. |
| [demo-api/src/demo_api/users.py](../demo-api/src/demo_api/users.py) | Пользователи: сервис и маршруты `/users`. |
| [demo-api/src/demo_api/auth.py](../demo-api/src/demo_api/auth.py) | Вход: сервис и маршрут `/login` (сюда агент добавляет rate limit). |
| [demo-api/src/demo_api/db.py](../demo-api/src/demo_api/db.py) | SQLite и схема БД (изменения требуют согласия). |
| [demo-api/src/demo_api/security.py](../demo-api/src/demo_api/security.py) | Хеширование паролей и токены (изменения требуют согласия). |
| [demo-api/tests/](../demo-api/tests/) | Тесты приложения (`conftest.py` — фикстуры, `test_api.py` — тесты API). |
| [demo-api/docs/api.md](../demo-api/docs/api.md) | Документация API (агент обновляет при изменениях). |
| [demo-api/pyproject.toml](../demo-api/pyproject.toml) | Зависимости и настройки ruff/mypy/pytest demo-api. |

## 10. Тесты jev-agent (`tests/`)

| Файл | Что проверяет |
|---|---|
| [scripted_model.py](../tests/scripted_model.py) | Сценарная модель: проигрывает заранее заданные ответы, чтобы гонять граф без сети. |
| [test_graph.py](../tests/test_graph.py) | Весь граф: успешный путь, repair, review, эскалация, правила, решения Jev, сжатие контекста, харнесс. |
| [test_policy.py](../tests/test_policy.py) | Разбор правил, сопоставление путей, deny / approval, защита тестов, привязка review к diff. |
| [test_workspace_tools.py](../tests/test_workspace_tools.py) | Изоляция копии, выход за пределы, инструменты, `run_check`, защита синтаксиса. |
| [test_proof.py](../tests/test_proof.py) | Доказательство регрессии на настоящем pytest. |
| [test_resilient_llm.py](../tests/test_resilient_llm.py) | Потоки, запасные модели, таймауты, понижение, починка вызовов инструментов. |
| [test_jev_client.py](../tests/test_jev_client.py) | Клиент Jev через `TypeSafeClassifier`: формат, повторы, трассировка. |
| [test_fabric.py](../tests/test_fabric.py) | Каждое решение Jev: уверенно, неуверенно, ошибка Jev. |
| [test_pr.py](../tests/test_pr.py) | PR через локальный bare-remote, описание PR. |
| [test_mcp_server.py](../tests/test_mcp_server.py) | MCP-сервер: инструменты и правила для внешнего клиента. |
| [test_scaffold.py](../tests/test_scaffold.py) | `init`: определение стека, сгенерированный AGENTS.md понимает policy engine. |
| [test_evals.py](../tests/test_evals.py) | Оценка: строки результатов, продолжение серии, сбой провайдера, вред, diff-проверки. |
| [test_cli.py](../tests/test_cli.py), [test_bench.py](../tests/test_bench.py), [test_config_and_llm.py](../tests/test_config_and_llm.py), [test_observability.py](../tests/test_observability.py) | Команды CLI, замер моделей, настройки, трассировка. |

## 11. Конфигурация, документация, CI

| Файл | За что отвечает |
|---|---|
| [AGENTS.md](../AGENTS.md) | Правила для агентов, которые **разрабатывают** jev-agent (не путать с `demo-api/AGENTS.md`). |
| [CLAUDE.md](../CLAUDE.md) | Для Claude Code: `@AGENTS.md` плюс про hook'и. |
| [.claude/settings.json](../.claude/settings.json) | Claude Code: запрет на `.env` и hook, который прогоняет ruff после каждой правки Python-файла. |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | CI на GitHub: ruff, mypy, тесты с порогом покрытия — отдельно для jev-agent и demo-api. |
| [pyproject.toml](../pyproject.toml) | Зависимости jev-agent и настройки ruff / mypy strict / pytest. |
| [.env.example](../.env.example) | Шаблон `.env`: ключи и цепочки моделей (сам `.env` в git не попадает). |
| [.gitignore](../.gitignore), [.python-version](../.python-version) | Игнорируемые файлы (`.env`, `runs/`, кеши); версия Python 3.12. |
| [README.md](../README.md) | Описание, установка, команды, подключение MCP, статус этапов. |
| [PROJECT_IDEA.md](../PROJECT_IDEA.md) | Исходная идея и план этапов M1–M7. |
| [docs/demo.md](demo.md) | Сценарий демо на 5 минут и ответы на вероятные вопросы. |
| [docs/project-map.md](project-map.md) | Этот файл. |

## Что не в git

- `.env` — ключи (NVIDIA, TypeSafe, LangSmith).
- `runs/<id>/` — результаты прогонов: `repo/` (копия с изменениями), `changes.diff`, `report.json` (план, проверки, доказательство регрессии, review, журнал правил, решения Jev, метрики, ссылка на трассу).
