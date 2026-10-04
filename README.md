# dinov3-embedder

Библиотека для получения frozen-эмбеддингов DINOv3 ViT из **исходных изображений
и bbox**. Один размеченный объект даёт один вектор. ID, метки, группы и геометрия
preprocessing сохраняются для проверки в соседнем пакете `embeddings-quality`.

## Установка

Нужны Python 3.10+, PyTorch и torchvision для вашего CPU/CUDA. Установите их
по [официальной инструкции PyTorch](https://pytorch.org/get-started/locally/), затем:

```bash
python -m pip install -e ./dinov3-embedder
python -m pip install -e './embeddings-quality[plots]'
```

Это команды из общей папки `digital`. Из самой папки `dinov3-embedder`
используйте `python -m pip install -e .`.

Backend — Hugging Face `AutoModel` + `AutoImageProcessor`. По умолчанию
`facebook/dinov3-vits16-pretrain-lvd1689m`: ViT-S/16, 384 координаты для CLS.
В первой версии поддерживаются **ViT**-чекпойнты DINOv3; ConvNeXt имеет другой
формат признаков. Определение токенов и модели описаны в
[документации Transformers](https://huggingface.co/docs/transformers/model_doc/dinov3).

Доступ к официальным весам нужно получить на
[странице модели](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
После получения доступа авторизуйтесь через `hf auth login` либо задайте
`HF_TOKEN` в окружении. При первом запуске Transformers загружает модель.
API также принимает `token=...`; токен не записывается в артефакты.

Локальная модель задаётся через `model_id="/path/to/hf-model"` и
`local_files_only=True`. Нужна директория формата Hugging Face с config, весами
и image processor; одиночный `.pth` из репозитория Meta этим backend не загружается.

## Python: исходное изображение и bbox

```python
from dinov3_embedder import DINOv3Embedder, EmbedderConfig, ImageSample

embedder = DINOv3Embedder(
    EmbedderConfig(
        device="auto",  # CUDA при доступности, иначе CPU
        batch_size=16,
        image_size=512,
        resize_mode="letterbox",  # aspect-preserving resize + padding
        pooling="cls",
    )
)

samples = [
    ImageSample(
        sample_id="qdrant-point-101",
        image="images/metal_001.jpg",
        bbox=(100, 80, 260, 140),  # x1,y1,x2,y2 в абсолютных пикселях
        label="scratch",
        group="metal_001",  # один image_id для всех bbox этого изображения
    ),
    ImageSample(
        sample_id="qdrant-point-102",
        image="images/metal_001.jpg",
        bbox=(300, 100, 390, 200),
        label="pitting",
        group="metal_001",
    ),
]

result = embedder.embed_samples(samples)
print(result.embeddings.shape)  # (2, 384)
result.save("outputs/dinov3.npz")
```

Для массивов без аннотаций:

```python
# Несколько объектов одного исходного изображения:
X = embedder.encode_bboxes(image, bboxes_xyxy)

# По одному bbox на изображение; допускается None для полного изображения:
X = embedder.encode(image_paths, bboxes=bboxes_xyxy)

# Готовые кропы или целые изображения:
X = embedder.encode(crops)
```

Изображение — локальный путь, PIL.Image или uint8 NumPy H×W, H×W×1,
H×W×3 (RGB), H×W×4 (RGBA). OpenCV BGR нужно сначала перевести в RGB.
Float-массивы отклоняются, чтобы диапазон пикселей не угадывался.
Входные изображения не изменяются. EXIF-ориентация автоматически не применяется:
координаты bbox относятся к пикселям декодированного файла.

Итерируемые входы обрабатываются батчами. Выход — NumPy float32 `(N, D)`
в порядке входных объектов. Все векторы остаются в RAM; изображения декодируются
для текущего батча. Один и тот же путь внутри батча декодируется один раз.
Неверный bbox, битый файл или повтор ID вызывают ошибку; строки не пропускаются.
Метки и группы должны быть заданы для всех объектов либо отсутствовать у всех.

## COCO JSON

```python
from dinov3_embedder import load_coco

samples = load_coco("annotations/train.json", image_root="images/train")
result = embedder.embed_samples(samples)
result.save("outputs/dinov3_train.npz")
```

COCO bbox `[x, y, width, height]` переводится в xyxy. `annotation.id` становится
sample_id, `category.name` — меткой, `image.id` — группой. Порядок annotations
сохраняется; записи `iscrowd` включаются. Для исключения crowd-объектов фильтруйте
annotations до загрузки. Изображения без annotations не создают bbox-эмбеддинги.

Для совпадения с Qdrant ID используйте CSV со своими point ID либо создайте
`ImageSample` из вашего объекта разметки. При сравнении SimCLR и DINOv3 каждый ID
должен обозначать один и тот же bbox.

## CSV и CLI

```csv
sample_id,image_path,label,group,x1,y1,x2,y2
point-101,metal_001.jpg,scratch,metal_001,100,80,260,140
point-102,metal_001.jpg,pitting,metal_001,300,100,390,200
point-103,metal_002.jpg,crack,metal_002,20,30,70,160
```

Обязательны `sample_id` и `image_path`. `label`, `group`, bbox необязательны.
Без bbox используется полное изображение. Если bbox задан, нужны все четыре
координаты. По умолчанию group — полный путь к исходному изображению.
Относительные пути разрешаются относительно CSV/JSON либо `image_root`.

```bash
dinov3-embed objects.csv --image-root images \
  --output outputs/dinov3.npz --image-size 512 --pooling cls

dinov3-embed annotations/train.json --image-root images/train \
  --output outputs/dinov3_train.npz --batch-size 8 --device cuda

python -m dinov3_embedder objects.csv --output outputs/dinov3.npz
```

Формат JSON по умолчанию трактуется как COCO; можно задать `--format csv|coco`.
Другие настройки: `--model`, `--revision`, `--dtype`, `--resize-mode`,
`--context-fraction`, `--no-normalize`, `--local-files-only`, `--cache-dir`.
CPU использует float32. На CUDA можно явно выбрать float16/bfloat16,
если это поддерживается устройством. Если не хватает GPU-памяти, уменьшите batch_size.

## Preprocessing и pooling

Обработка: RGB → bbox crop → необязательный context → resize/padding →
rescale и channel normalization из image processor модели. Дополнительный
resize/center crop в Transformers отключён. Модель работает в eval и
`torch.inference_mode()`, параметры заморожены.

По умолчанию кроп вписывается в 512×512 с сохранением aspect ratio, чёрный
padding располагается по центру. Это настройка эксперимента для сравнения
с вашим SimCLR; она не повторяет стандартный preprocessing модели при 224×224.
Режимы:

- `letterbox`: сохранение пропорций и padding, bicubic resize.
- `stretch`: прямой resize до квадрата без padding, bicubic resize.
- `none`: отсутствие resize; исходный кроп уже должен иметь image_size×image_size.

`context_fraction=0.2` добавляет **20% ширины/высоты bbox с каждой стороны**
(итого до 1.4 исходного размера). Контекст обрезается границами изображения.
Сам bbox должен целиком лежать внутри изображения и иметь положительную площадь.
Дробные координаты округляются наружу до целых пикселей.
Размер входа должен быть кратен patch_size модели; это проверяется после загрузки.

| Pooling | Вектор | Размер ViT-S/16 |
|---|---|---:|
| `cls` | CLS-токен последнего нормализованного слоя | 384 |
| `patch_mean` | Среднее patch-токенов без CLS/register-токенов | 384 |
| `cls_patch_mean` | Конкатенация CLS и среднего patch-токенов | 768 |

По умолчанию итоговый вектор L2-нормализуется; `normalize=False` сохраняет
ненормализованный результат pooling. Конкатенация нормализуется целиком.
**Среднее patch-токенов включает области padding.** CLS также учитывает весь
вход модели. Если padding влияет на качество, сравните letterbox/stretch,
context и nuisance probes на одних объектах. Структура токенов описана в
[официальной документации](https://huggingface.co/docs/transformers/model_doc/dinov3).

## Выход и проверка качества

`result.save("dinov3.npz")` создаёт:

- `dinov3.npz`: embeddings, sample_ids, labels и groups без pickle.
- `dinov3.metadata.csv`: размеры исходного изображения/bbox/crop, aspect_ratio,
  доля bbox, фактический resize_scale_x/y и padding_fraction по sample_id.
- `dinov3.config.json`: model/revision, pooling, обработка, размерность,
  устройство, параметры image processor и версии библиотек.

Метки/groups сохраняются, если были заданы. Повторный save перезаписывает эти
файлы. Для воспроизводимого checkpoint задайте `revision`/`--revision` как commit
модели; фактически разрешённый commit также сохраняется, если Transformers его сообщает.

```bash
embeddings-quality outputs/dinov3.npz --output reports/dinov3 --plots \
  --metadata outputs/dinov3.metadata.csv
```

Или напрямую:

```python
from embeddings_quality import evaluate_embeddings

report = evaluate_embeddings(
    result.embeddings,
    result.labels,
    sample_ids=result.sample_ids,
    groups=result.groups,
    metadata={
        "padding_fraction": [row["padding_fraction"] for row in result.metadata],
        "bbox_area_fraction": [row["bbox_area_fraction"] for row in result.metadata],
    },
)
report.save("reports/dinov3", plots=True)
```

Для сравнения с SimCLR используйте одинаковые bbox, метки и группы, а также
одинаковые параметры аудита. Пространства могут иметь разные D: векторы
сравниваются через оценки качества, а не покоординатно. Проверьте соответствие
sample_ids, если выгрузки имеют разный порядок. Если меняете одновременно
backbone, pooling и preprocessing, результат относится к полному pipeline.

Пример полного запуска — [examples/embed_and_audit.py](examples/embed_and_audit.py).

## Проверка библиотеки

Ruff проверяет и форматирует `src`, тесты и примеры. Mypy проверяет `src`;
аннотированы изображения, bbox, массивы признаков и результаты экспорта.
Конфигурация — в `pyproject.toml`. Динамический интерфейс Transformers
проверяется на границе с типизированным кодом; его внутренние модули mypy
не анализирует. `py.typed` включает проверку аннотаций пакета у пользователей.

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m mypy
HF_HUB_OFFLINE=1 python -m pytest -q
python -m build
```

Для применения форматирования используйте `python -m ruff format .`.

Тесты выполняют настоящий forward DINOv3 из Transformers с маленьким локальным
чекпойнтом и случайными весами. Проверяются кропы, порядок объектов, pooling,
исключение register-токенов, батчи, нормализация, COCO/CSV, CLI и совместимость NPZ
с `embeddings-quality` (если установлен). Они не оценивают pretrained representation.
Веса Meta и пользовательский датасет в тестах не загружаются.
