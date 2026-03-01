#!/bin/bash

if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python 3 не найден. Установите: sudo apt install python3 python3-pip"
    exit 1
fi

if ! python3 -m pip --version &> /dev/null 2>&1; then
    echo "Ошибка: pip не найден. Установите: sudo apt install python3-pip"
    exit 1
fi

echo "Подготовка библиотек..."

if [ ! -f "venv/bin/activate" ]; then
    rm -rf venv
    VENV_OUTPUT=$(python3 -m venv venv 2>&1)
    if [ $? -ne 0 ]; then
        echo "Ошибка: не удалось создать виртуальное окружение."
        echo "Установите: sudo apt install python3-venv"
        echo ""
        echo "Детали ошибки:"
        echo "$VENV_OUTPUT"
        exit 1
    fi
fi

source venv/bin/activate

PIP_OUTPUT=$(pip install --quiet numpy pandas scikit-learn 2>&1)
if [ $? -ne 0 ]; then
    echo "Ошибка при установке библиотек:"
    echo "$PIP_OUTPUT"
    exit 1
fi

echo "Запуск: data_creation.py"
python3 data_creation.py > /dev/null 2>&1

echo "Запуск: data_preprocessing.py"
python3 data_preprocessing.py > /dev/null 2>&1

echo "Запуск: model_preparation.py"
python3 model_preparation.py > /dev/null 2>&1

echo "Запуск: model_testing.py"
python3 model_testing.py






