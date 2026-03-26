#!/usr/bin/env python3
"""
Парсинг TSF файлов (Monash Time Series Repository).

Формат TSF:
- @meta-information строки в начале
- Данные: series_name,start_timestamp,val1:val2:val3:...

Запуск:
    python tests/parse_tsf.py --input real_datasets/weather_dataset.tsf
"""

import numpy as np
from pathlib import Path
import argparse

# Директория для данных
DATA_DIR = Path(__file__).parent.parent / "real_datasets"


def parse_tsf_file(filepath, n_series=5, series_length=500):
    """
    Парсинг TSF файла и извлечение временных рядов.
    
    Формат TSF: series_name:variable_name,val1,val2,val3,...
    
    Args:
        filepath: Путь к TSF файлу
        n_series: Число рядов для извлечения
        series_length: Длина каждого ряда
        
    Returns:
        List[np.ndarray]: Список временных рядов
    """
    print(f"📊 Парсинг TSF: {filepath.name}")
    
    series_list = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Пропускаем meta-информацию (начинается с @)
    data_lines = [line for line in lines 
                  if not line.startswith('@') and line.strip()]
    
    count = 0
    for line in data_lines:
        if count >= n_series:
            break
        
        # Формат: series_name:variable_name,val1,val2,val3,...
        if ':' not in line:
            continue
        
        parts = line.strip().split(':')
        if len(parts) < 2:
            continue
        
        # Извлекаем значения (после второго ':')
        data_part = ':'.join(parts[1:])  # variable_name,val1,val2,...
        values_str = data_part.split(',')
        
        values = []
        for val_str in values_str[1:]:  # Пропускаем имя переменной
            try:
                val = float(val_str.strip())
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
            except ValueError:
                continue
        
        if len(values) > 100:
            series = np.array(values[:series_length])
            series_list.append(series)
            count += 1
            print(f"   ✅ Ряд {count}: len={len(series)}")
    
    return series_list


def save_series(name, series_list, series_length=500):
    """Сохранение рядов в формате npy."""
    saved_dir = DATA_DIR / "extracted"
    saved_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    for i, series in enumerate(series_list):
        series = np.array(series)[:series_length]
        if len(series) > 100:
            save_path = saved_dir / f"{name}_series_{i}.npy"
            np.save(save_path, series)
            print(f"   ✅ {name}_series_{i}.npy (len={len(series)})")
            saved_count += 1
    
    return saved_count


def parse_weather_dataset():
    """Парсинг weather датасета."""
    print("📊 Обработка Weather dataset...")
    
    tsf_path = DATA_DIR / "weather_dataset.tsf"
    
    if not tsf_path.exists():
        print(f"   ❌ Файл не найден: {tsf_path}")
        return []
    
    series_list = parse_tsf_file(tsf_path, n_series=5, series_length=500)
    
    if series_list:
        save_series("weather", series_list)
    
    return series_list


def parse_traffic_dataset():
    """Парсинг traffic датасета."""
    print("📊 Обработка Traffic dataset...")
    
    tsf_path = DATA_DIR / "traffic_dataset.tsf"
    
    if not tsf_path.exists():
        print(f"   ❌ Файл не найден: {tsf_path}")
        return []
    
    series_list = parse_tsf_file(tsf_path, n_series=5, series_length=500)
    
    if series_list:
        save_series("traffic", series_list)
    
    return series_list


def parse_exchange_dataset():
    """Парсинг exchange rate датасета."""
    print("📊 Обработка Exchange Rate dataset...")
    
    tsf_path = DATA_DIR / "exchange_rate_dataset.tsf"
    
    if not tsf_path.exists():
        print(f"   ❌ Файл не найден: {tsf_path}")
        return []
    
    series_list = parse_tsf_file(tsf_path, n_series=8, series_length=500)
    
    if series_list:
        save_series("exchange", series_list)
    
    return series_list


def parse_all_tsf_files(n_series=5, series_length=500):
    """Парсинг всех TSF файлов в директории real_datasets."""
    print("=" * 60)
    print("📊 Массовый парсинг всех TSF файлов")
    print("=" * 60)
    
    tsf_files = list(DATA_DIR.glob("*.tsf"))
    
    if not tsf_files:
        print("❌ TSF файлы не найдены в директории real_datasets/")
        return
    
    print(f"📁 Найдено файлов: {len(tsf_files)}")
    print()
    
    total_series = 0
    for tsf_file in tsf_files:
        dataset_name = tsf_file.stem.replace('_dataset', '').replace('_with_missing_values', '')
        print(f"\n{'='*60}")
        print(f"📊 Обработка: {tsf_file.name}")
        print(f"{'='*60}")
        
        series_list = parse_tsf_file(
            tsf_file,
            n_series=n_series,
            series_length=series_length
        )
        
        if series_list:
            saved_count = save_series(dataset_name, series_list, series_length)
            total_series += saved_count
    
    print("\n" + "=" * 60)
    print(f"✅ ВСЕГО СОХРАНЕНО: {total_series} рядов из {len(tsf_files)} файлов")
    print("=" * 60)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Парсинг TSF файлов Monash Repository")
    parser.add_argument("--input", type=str, 
                        help="Путь к TSF файлу")
    parser.add_argument("--n-series", type=int, default=5,
                        help="Число рядов для извлечения")
    parser.add_argument("--length", type=int, default=500,
                        help="Длина ряда")
    parser.add_argument("--name", type=str, default="dataset",
                        help="Имя датасета для сохранения")
    parser.add_argument("--all", action="store_true",
                        help="Парсить все TSF файлы в директории")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 Парсинг TSF файлов")
    print("=" * 60)
    
    if args.input:
        # Парсинг указанного файла
        filepath = Path(args.input)
        if not filepath.exists():
            print(f"❌ Файл не найден: {filepath}")
            return
        
        series_list = parse_tsf_file(
            filepath, 
            n_series=args.n_series,
            series_length=args.length
        )
        
        if series_list:
            save_series(args.name, series_list)
    elif args.all:
        # Массовый парсинг всех файлов
        parse_all_tsf_files(
            n_series=args.n_series,
            series_length=args.length
        )
    else:
        # Парсинг всех доступных датасетов
        parse_weather_dataset()
        parse_traffic_dataset()
        parse_exchange_dataset()
    
    print("\n" + "=" * 60)
    print("✅ Готово")
    print("=" * 60)


if __name__ == "__main__":
    main()
