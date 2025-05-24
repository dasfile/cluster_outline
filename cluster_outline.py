# MIT License

# Copyright (c) 2024–present Ivan Rodionov, ChatGPT, Claude and Contributors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# The Software makes use of insights generated with the assistance of Grok
# (https://x.ai/grok), a large language model developed by xAI, and Claude
# (https://claude.ai), a large language model developed by Anthropic.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Cluster DXF Outline
=====================

The application is designed for automatic construction of the external outline of equipment based on DXF drawings. 

Use case: plan simplification, creating outlines for specifications, area calculations or visual diagrams. 

Install dependencies:
    pip install ezdxf opencv-python scikit-learn tqdm numpy 

Also, if you are using Windows 10 Pro N, you may need the Media Feature Pack. You can install it using this command:
    DISM /Online /Add-Capability /CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0 

Run:
    python cluster_outline.py input.dxf 

The result is saved to file input_contour.dxf with a new layer OUTLINE_EDGES containing the detected outlines. 

Приложение предназначено для автоматического построения внешнего контура оборудования по DXF-чертежам. 

Сценарий использования: упрощение планов, построение обводок для экспликаций, расчётов площадей или визуальных схем.

Установка зависимостей:
    pip install ezdxf opencv-python scikit-learn tqdm numpy

Возможно, так же, если у Вас Windows 10 Pro N, вам потребуется MediaFeaturePack. Установить можно так:
    DISM /Online /Add-Capability /CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0

Запуск:
    python cluster_outline.py input.dxf

Результат сохраняется в файл input_contour.dxf с новым слоем OUTLINE_EDGES, содержащим найденные контуры.

"""

import ezdxf
import sys
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from multiprocessing import Pool, cpu_count, get_context
from datetime import datetime, timedelta

# ====== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ======
IMG_SIZE = 4096               # Размер изображения для растеризации
LINE_THICKNESS = 2            # Толщина линий на изображении
OUT_LAYER = "OUTLINE_EDGES"   # Имя слоя для обводки
EPS = 1300                    # Радиус для DBSCAN кластеризации
MIN_SAMPLES = 3               # Минимальное число примитивов в кластере
USE_PARALLEL = False           # Включить/выключить параллельную обработку, False/True
# =====================================

def format_time(seconds):
    """Форматирование времени в читаемый вид."""
    if seconds < 60:
        return f"{seconds:.2f} секунд"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} мин {secs:.2f} сек"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} ч {minutes} мин {secs:.2f} сек"

def explode_all(modelspace):
    """Рекурсивно взорвать все блоки и составные примитивы."""
    max_iterations = 10  # Защита от бесконечной рекурсии
    iteration = 0
    
    while iteration < max_iterations:
        # Найти все объекты, которые можно взорвать
        to_explode = list(modelspace.query('INSERT LWPOLYLINE POLYLINE DIMENSION HATCH MTEXT'))
        
        if not to_explode:
            break
            
        print(f"Итерация {iteration + 1}: найдено {len(to_explode)} объектов для взрыва")
        
        exploded_count = 0
        for entity in tqdm(to_explode, desc=f"Взрыв объектов (итерация {iteration + 1})"):
            try:
                if hasattr(entity, 'explode'):
                    entity.explode()
                    exploded_count += 1
                elif entity.dxftype() == 'INSERT':
                    # Дополнительная обработка для блоков INSERT
                    entity.explode()
                    exploded_count += 1
            except Exception as e:
                # Игнорируем ошибки взрыва отдельных объектов
                continue
        
        if exploded_count == 0:
            break
            
        print(f"Взорвано объектов: {exploded_count}")
        iteration += 1
    
    print(f"Взрыв завершён за {iteration} итераций")

def get_entity_center(entity):
    """Получить центр объекта с поддержкой дополнительных типов."""
    try:
        if entity.dxftype() == 'LINE':
            p1 = entity.dxf.start
            p2 = entity.dxf.end
            return [(p1.x + p2.x) / 2, (p1.y + p2.y) / 2]
        elif entity.dxftype() in ('CIRCLE', 'ARC'):
            c = entity.dxf.center
            return [c.x, c.y]
        elif entity.dxftype() == 'POINT':
            p = entity.dxf.location
            return [p.x, p.y]
        elif entity.dxftype() == 'TEXT':
            p = entity.dxf.insert
            return [p.x, p.y]
        elif entity.dxftype() == 'POLYLINE':
            # Для полилинии берём центр ограничивающего прямоугольника
            points = list(entity.points())
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]
        elif hasattr(entity.dxf, 'insert'):
            # Для объектов с точкой вставки
            p = entity.dxf.insert
            return [p.x, p.y]
        elif hasattr(entity.dxf, 'center'):
            # Для объектов с центром
            c = entity.dxf.center
            return [c.x, c.y]
    except Exception:
        pass
    
    return [0, 0]

def get_bounds(entities):
    """Получить границы объектов с поддержкой дополнительных типов."""
    xs, ys = [], []
    
    for e in entities:
        try:
            if e.dxftype() == 'LINE':
                xs += [e.dxf.start.x, e.dxf.end.x]
                ys += [e.dxf.start.y, e.dxf.end.y]
            elif e.dxftype() in ('CIRCLE', 'ARC'):
                c = e.dxf.center
                r = e.dxf.radius
                xs += [c.x - r, c.x + r]
                ys += [c.y - r, c.y + r]
            elif e.dxftype() == 'POINT':
                p = e.dxf.location
                xs.append(p.x)
                ys.append(p.y)
            elif e.dxftype() == 'TEXT':
                p = e.dxf.insert
                xs.append(p.x)
                ys.append(p.y)
            elif e.dxftype() == 'POLYLINE':
                points = list(e.points())
                if points:
                    xs += [p[0] for p in points]
                    ys += [p[1] for p in points]
            elif hasattr(e.dxf, 'insert'):
                p = e.dxf.insert
                xs.append(p.x)
                ys.append(p.y)
        except Exception:
            continue
    
    if not xs or not ys:
        return 0, 100, 0, 100  # Значения по умолчанию
    
    return min(xs), max(xs), min(ys), max(ys)

def draw_entities_to_image(entities, bounds):
    """Отрисовка объектов на изображение с поддержкой дополнительных типов."""
    xmin, xmax, ymin, ymax = bounds
    scale = IMG_SIZE / max(xmax - xmin, ymax - ymin)
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    for e in entities:
        try:
            if e.dxftype() == 'LINE':
                p1 = e.dxf.start
                p2 = e.dxf.end
                x1 = int((p1.x - xmin) * scale)
                y1 = IMG_SIZE - int((p1.y - ymin) * scale)
                x2 = int((p2.x - xmin) * scale)
                y2 = IMG_SIZE - int((p2.y - ymin) * scale)
                cv2.line(img, (x1, y1), (x2, y2), 0, LINE_THICKNESS)
            elif e.dxftype() == 'CIRCLE':
                c = e.dxf.center
                r = int(e.dxf.radius * scale)
                x = int((c.x - xmin) * scale)
                y = IMG_SIZE - int((c.y - ymin) * scale)
                cv2.circle(img, (x, y), r, 0, LINE_THICKNESS)
            elif e.dxftype() == 'ARC':
                # Приближение дуги линиями
                c = e.dxf.center
                r = e.dxf.radius
                start_angle = np.radians(e.dxf.start_angle)
                end_angle = np.radians(e.dxf.end_angle)
                
                # Количество сегментов для аппроксимации дуги
                segments = max(8, int(abs(end_angle - start_angle) * r * scale / 10))
                
                if end_angle < start_angle:
                    end_angle += 2 * np.pi
                
                angles = np.linspace(start_angle, end_angle, segments)
                points = []
                
                for angle in angles:
                    x = int((c.x + r * np.cos(angle) - xmin) * scale)
                    y = IMG_SIZE - int((c.y + r * np.sin(angle) - ymin) * scale)
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i + 1], 0, LINE_THICKNESS)
            elif e.dxftype() == 'POLYLINE':
                points = list(e.points())
                if len(points) > 1:
                    img_points = []
                    for p in points:
                        x = int((p[0] - xmin) * scale)
                        y = IMG_SIZE - int((p[1] - ymin) * scale)
                        img_points.append((x, y))
                    
                    for i in range(len(img_points) - 1):
                        cv2.line(img, img_points[i], img_points[i + 1], 0, LINE_THICKNESS)
                    
                    # Если полилиния замкнутая, соединяем последнюю точку с первой
                    if e.is_closed:
                        cv2.line(img, img_points[-1], img_points[0], 0, LINE_THICKNESS)
            elif e.dxftype() == 'POINT':
                p = e.dxf.location
                x = int((p.x - xmin) * scale)
                y = IMG_SIZE - int((p.y - ymin) * scale)
                cv2.circle(img, (x, y), 2, 0, -1)  # Маленькая заливка для точки
        except Exception:
            continue

    return img, scale, xmin, ymin

def extract_outer_contour(img):
    img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours_to_dxf(doc, contours, scale, xmin, ymin):
    msp = doc.modelspace()
    if OUT_LAYER not in doc.layers:
        doc.layers.add(name=OUT_LAYER, color=1)

    for cnt in contours:
        pts = cnt.squeeze()
        if len(pts.shape) != 2:
            continue
        for i in range(len(pts) - 1):
            x1 = pts[i][0] / scale + xmin
            y1 = (IMG_SIZE - pts[i][1]) / scale + ymin
            x2 = pts[i + 1][0] / scale + xmin
            y2 = (IMG_SIZE - pts[i + 1][1]) / scale + ymin
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": OUT_LAYER})

def process_cluster(args):
    label, entities, labels = args
    if label == -1:
        return []
    cluster_entities = [e for e, l in zip(entities, labels) if l == label]
    bounds = get_bounds(cluster_entities)
    img, scale, xmin, ymin = draw_entities_to_image(cluster_entities, bounds)
    contours = extract_outer_contour(img)
    return (contours, scale, xmin, ymin)

def main():
    # Начало измерения времени
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"Начало работы: {start_datetime.strftime('%H:%M:%S')}")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Укажите имя DXF-файла: python script.py input.dxf")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_name(input_path.stem + "_contour.dxf")

    # Загрузка файла
    load_start = time.time()
    print(f"Загрузка {input_path.name}...")
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()
    load_time = time.time() - load_start
    print(f"Загрузка завершена за {format_time(load_time)}")
    
    # Взрыв блоков
    explode_start = time.time()
    explode_all(msp)
    explode_time = time.time() - explode_start
    print(f"Взрыв блоков завершён за {format_time(explode_time)}")

    # Подготовка данных
    prep_start = time.time()
    # Расширенный список поддерживаемых типов объектов
    supported_types = ('LINE', 'CIRCLE', 'ARC', 'POLYLINE', 'LWPOLYLINE', 'POINT', 'TEXT')
    entities = [e for e in msp if e.dxftype() in supported_types]
    centers = np.array([get_entity_center(e) for e in entities])
    prep_time = time.time() - prep_start
    print(f"Подготовка данных ({len(entities)} объектов) за {format_time(prep_time)}")

    # Кластеризация
    cluster_start = time.time()
    clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(centers)
    labels = clustering.labels_
    unique_labels = list(set(labels))
    cluster_time = time.time() - cluster_start
    print(f"Кластеризация ({len(unique_labels)} кластеров) за {format_time(cluster_time)}")

    # Обработка кластеров
    process_start = time.time()
    try:
        if USE_PARALLEL:
            with get_context("spawn").Pool(cpu_count(), maxtasksperchild=1) as pool:
                results = list(tqdm(
                    pool.imap(process_cluster, [(label, entities, labels) for label in unique_labels]),
                    total=len(unique_labels),
                    desc="Обработка кластеров"
                ))
        else:
            raw_tasks = [(label, entities, labels) for label in unique_labels]
            results = list(tqdm(map(process_cluster, raw_tasks), total=len(raw_tasks), desc="Обработка кластеров"))
    except KeyboardInterrupt:
        print("\nПрервано пользователем. Завершение работы...")
        sys.exit(1)
    
    process_time = time.time() - process_start
    print(f"Обработка кластеров завершена за {format_time(process_time)}")

    # Создание DXF контуров
    dxf_start = time.time()
    contour_count = 0
    for result in results:
        if not result:
            continue
        contours, scale, xmin, ymin = result
        draw_contours_to_dxf(doc, contours, scale, xmin, ymin)
        contour_count += len(contours)
    dxf_time = time.time() - dxf_start
    print(f"Создание {contour_count} контуров в DXF за {format_time(dxf_time)}")

    # Сохранение файла
    save_start = time.time()
    doc.saveas(output_path)
    save_time = time.time() - save_start
    print(f"Сохранение файла за {format_time(save_time)}")

    # Общее время работы
    total_time = time.time() - start_time
    end_datetime = datetime.now()
    
    print("=" * 50)
    print("ОТЧЁТ О ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
    print(f"Начало:              {start_datetime.strftime('%H:%M:%S')}")
    print(f"Окончание:           {end_datetime.strftime('%H:%M:%S')}")
    print(f"Общее время:         {format_time(total_time)}")
    print("-" * 30)
    print(f"Загрузка файла:      {format_time(load_time)} ({load_time/total_time*100:.1f}%)")
    print(f"Взрыв блоков:        {format_time(explode_time)} ({explode_time/total_time*100:.1f}%)")
    print(f"Подготовка данных:   {format_time(prep_time)} ({prep_time/total_time*100:.1f}%)")
    print(f"Кластеризация:       {format_time(cluster_time)} ({cluster_time/total_time*100:.1f}%)")
    print(f"Обработка кластеров: {format_time(process_time)} ({process_time/total_time*100:.1f}%)")
    print(f"Создание DXF:        {format_time(dxf_time)} ({dxf_time/total_time*100:.1f}%)")
    print(f"Сохранение:          {format_time(save_time)} ({save_time/total_time*100:.1f}%)")
    print("=" * 50)
    print(f"Сохранено: {output_path.name}")

if __name__ == "__main__":
    main()