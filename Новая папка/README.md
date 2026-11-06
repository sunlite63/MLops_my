{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 MLOps Assignment\n",
    "\n",
    "Решение домашних заданий по курсу MLOps. Проект демонстрирует полный цикл разработки ML-приложения: от настройки окружения до трекинга экспериментов.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📁 Структура проекта"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Дерево проекта:\n",
    "#\n",
    "# ├── .github/workflows/ci.yml    # GitHub Actions CI/CD\n",
    "# ├── .dvc/                        # Конфигурация DVC\n",
    "# ├── data/                        # Данные (версионируются DVC)\n",
    "# ├── output/                      # Артефакты и метрики\n",
    "# ├── src/\n",
    "# │   ├── __init__.py             # Базовые утилиты (ДЗ 1)\n",
    "# │   ├── generate_data.py        # DVC stage: генерация данных (ДЗ 3)\n",
    "# │   ├── process_data.py         # DVC stage: расчет метрик (ДЗ 3)\n",
    "# │   ├── models.py               # Нейросетевые архитектуры (ДЗ 4)\n",
    "# │   ├── dataset.py              # Датасеты и загрузчики (ДЗ 4)\n",
    "# │   ├── trainer.py              # Логика обучения (ДЗ 4)\n",
    "# │   ├── train.py                # Обучение с ручным MLflow (ДЗ 5)\n",
    "# │   └── train_autolog.py        # Обучение с автологгингом (ДЗ 5)\n",
    "# ├── tests/                      # Тесты (pytest + hypothesis)\n",
    "# ├── dvc.yaml                    # Пайплайн DVC\n",
    "# ├── pyproject.toml              # Конфигурация проекта и зависимостей\n",
    "# └── README.ipynb                # Этот файл"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ⚡ Быстрый старт"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Клонирование и установка зависимостей\n",
    "# !git clone <repo-url>\n",
    "# %cd mlops-assignment\n",
    "\n",
    "# Установка всех зависимостей через uv (выполнить в терминале):\n",
    "# uv sync --all-extras"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📋 ДЗ 1: Окружение и линтеры\n",
    "\n",
    "Используется `uv` для управления зависимостями.\n",
    "\n",
    "| Инструмент | Назначение |\n",
    "|---|---|\n",
    "| `ruff check` | Линтер (проверка стиля) |\n",
    "| `ruff format` | Форматтер (автоформатирование) |\n",
    "| `ty` | Статическая проверка типов |\n",
    "\n",
    "**Настройки:**\n",
    "- Длина строки: `100` символов\n",
    "- Правила: `E`, `F`, `I`, `UP`, `B`, `W`\n",
    "- Python: `3.11`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Команды для проверки (выполнять в терминале):\n",
    "# uv run ruff check .\n",
    "# uv run ruff format --check .\n",
    "# uv run ty check"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📋 ДЗ 2: Работа с Git\n",
    "\n",
    "Отработаны навыки:\n",
    "- ✅ Git-flow: `main` → `develop` → `feature/*`\n",
    "- ✅ Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`)\n",
    "- ✅ Разрешение merge conflicts\n",
    "- ✅ `.gitignore` для ML-артефактов"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📋 ДЗ 3: Версионирование данных (DVC)\n",
    "\n",
    "Пайплайн из двух стадий:\n",
    "\n",
    "```mermaid\n",
    "graph LR\n",
    "    A[generate_data.py] -->|data/raw_data.csv| B[process_data.py]\n",
    "    B --> C[output/metrics.json]\n",
    "```\n",
    "\n",
    "**Стадии:**\n",
    "1. `create_dataset` — генерирует синтетический CSV (100 строк, seed=42)\n",
    "2. `compute_metrics` — считает `total_samples` и `positive_ratio`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Инициализация DVC (терминал):\n",
    "# uv run dvc init\n",
    "# uv run dvc remote add -d local ~/.dvc_storage\n",
    "\n",
    "# Запуск пайплайна:\n",
    "# uv run dvc repro\n",
    "\n",
    "# Просмотр метрик:\n",
    "# uv run dvc metrics show\n",
    "# uv run dvc metrics diff"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📋 ДЗ 4: Тестирование (TDD, pytest, Hypothesis)\n",
    "\n",
    "### Пирамида тестирования:\n",
    "\n",
    "| Маркер | Описание | Пример |\n",
    "|---|---|---|\n",
    "| `unit` | Проверка отдельных компонентов | Форма выхода, длина датасета |\n",
    "| `property` | Инварианты на случайных данных | Форма при любых размерах (Hypothesis) |\n",
    "| `smoke` | Быстрые проверки адекватности | Loss конечен и убывает, градиенты есть |\n",
    "| `overfit` | Переобучение на одном батче | Loss < 0.05 за 200 шагов |\n",
    "| `e2e` | Сквозной тест пайплайна | Обучение → accuracy > 0.8 → инференс |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Запуск тестов (термина