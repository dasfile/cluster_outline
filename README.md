# cluster_outline

## Описание / Description

### Русский

Программа осуществляет обводку контуров элементов технологического оборудования, выгруженных с избыточной детализацией из САПР, в DXF файлах. Использует библиотеки Ezdxf, OpenCV в среде Python 3. 

Это необходимо при разработке архитектурно-строительных планов, когда чертежи, переданные технологами, содержат избыточную детализацию и затрудняют работу с чертежом.

**Как работает:**
- С помощью библиотеки Ezdxf программа взрывает все блоки и растеризует чертёж
- При помощи библиотеки машинного зрения OpenCV обнаруживает контуры оборудования и обводит их
- Линия обводки выносится на отдельный слой в результирующем файле
- Результирующий файл является копией исходного файла с добавленным слоем обводки

### English

The program creates outline contours for technological equipment elements exported with excessive detail from CAD systems in DXF files. It uses Ezdxf and OpenCV libraries in Python 3 environment.

This is necessary when developing architectural and construction plans, where drawings provided by process engineers contain excessive detail that complicates working with the drawing.

**How it works:**
- Using the Ezdxf library, the program explodes all blocks and rasterizes the drawing
- Using the OpenCV computer vision library, it detects equipment contours and outlines them
- The outline is placed on a separate layer in the resulting file
- The resulting file is a copy of the original file with an added outline layer

## Установка зависимостей / Dependencies Installation

### Русский

Убедитесь, что установлены все необходимые библиотеки Python:

```bash
pip install ezdxf opencv-python scikit-learn tqdm numpy
```

**Для пользователей Windows 10 Pro N:** возможно потребуется MediaFeaturePack:
```bash
DISM /Online /Add-Capability /CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0
```

### English

Make sure all required Python libraries are installed:

```bash
pip install ezdxf opencv-python scikit-learn tqdm numpy
```

**For Windows 10 Pro N users:** you may need MediaFeaturePack:
```bash
DISM /Online /Add-Capability /CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0
```

## Использование / Usage

### Русский

1. Поместите файл .dxf с оборудованием в ту же папку, что и скрипт `cluster_outline.py`
2. Откройте командную строку (например, CMD в Windows)
3. Перейдите в директорию со скриптом
4. Запустите команду:

```bash
python cluster_outline.py ваш_файл.dxf
```

При успешном выполнении вы получите результирующий файл в той же папке.

### English

1. Place the .dxf file with equipment in the same folder as the `cluster_outline.py` script
2. Open command line interface (e.g., CMD in Windows)
3. Navigate to the script directory
4. Run the command:

```bash
python cluster_outline.py your_file.dxf
```

Upon successful execution, you will get the resulting file in the same folder.

## Важные замечания / Important Notes

### Русский

- **Предварительная подготовка файла:** Вам необходимо вручную удалить ненужные элементы строительных конструкций и координационные оси. Должны остаться только элементы технологического оборудования
- **Размер файла:** Если файл очень большой (более 70 МБ), лучше разбить работу на части
- **Требования:** Python 3.x

### English

- **File preparation:** You need to manually remove unnecessary building structure elements and coordinate axes. Only technological equipment elements should remain
- **File size:** If the file is very large (over 70 MB), it's better to split the work into parts
- **Requirements:** Python 3.x

## Системные требования / System Requirements

- Python 3.x
- Windows/Linux/macOS
- Минимум 4 ГБ RAM для файлов среднего размера / Minimum 4 GB RAM for medium-sized files

## Лицензия / License

MIT
