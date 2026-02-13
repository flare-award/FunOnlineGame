# Local Game Autopilot (Python, Windows-compatible)

Локальный проект для экспериментов с компьютерным зрением и управлением игрой через горячие клавиши.

## Важно (этика и право)
**Используйте этот инструмент только для экспериментов и обучения в локальных условиях. Не используйте для получения нечестного преимущества в многопользовательских режимах; разработчик не несёт ответственности за нарушение правил сервисов.**

## Возможности
- Захват прямоугольной области экрана (`mss` + `opencv`).
- Лёгкая CNN-модель (`TinyCNN`) на PyTorch.
- Обучение локально на собственных данных (`record_data.py` + `train.py`).
- Экспорт модели в TorchScript и ONNX.
- GUI (Tkinter): ВКЛ/ВЫКЛ, статус, настройка хоткеев.
- Безопасный режим: при низкой уверенности не отправляет команды.
- Режим only-demo: полностью отключает отправку клавиш.
- Логирование в файл.
- Демонстрационный режим воспроизведения с визуализацией предсказаний.
- Unit-тесты для ключевых модулей.

## Структура проекта
```text
FunOnlineGame/
  autopilot_bot/
    __init__.py
    capture.py
    config.py
    controller.py
    gui.py
    model.py
    trainer.py
    utils.py
  configs/
    config.yaml
    config_wasd.yaml
    config_arrows.yaml
  data/
    demo_dataset/
      frames/
      labels/
  logs/
  models/
  tests/
    test_capture.py
    test_controller.py
    test_model_predict.py
  demo_playback.py
  record_data.py
  run.py
  train.py
  requirements.txt
  README.md
```

## Установка
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# или cmd:
.venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

## Быстрый запуск (smoke test)
1. Сгенерировать мини-датасет и обучить тестовую модель:
```bash
python train.py --bootstrap-demo --epochs 1
```
2. Запустить автопилот в demo-режиме (без нажатий клавиш):
```bash
python run.py --demo
```
3. Проверить визуализацию предсказаний на демо-данных:
```bash
python demo_playback.py --dataset data/demo_dataset
```

## Запись данных (ручная игра)
```bash
python record_data.py --config configs/config.yaml --out data/session_01 --seconds 60
```
Скрипт сохраняет кадры в `frames/` и `labels.csv` (метка определяется по текущей зажатой клавише из action map).

## Обучение на записанных данных
```bash
python train.py --dataset data/session_01 --epochs 5
```
Результат:
- TorchScript: `models/tinycnn_scripted.pt`
- ONNX: `models/tinycnn.onnx`

## Реальный запуск
```bash
python run.py --config configs/config.yaml
```
- Нажмите кнопку **ВКЛ** в GUI.
- Если `only_demo: true`, клавиши отправляться не будут.
- Для реальных экспериментов локально: `only_demo: false` и желательно `game_window_title_contains`.

## Ключевые настройки (`configs/config.yaml`)
- `capture_region`: область экрана.
- `capture_fps`, `inference_fps`: частоты записи/инференса.
- `action_list`, `action_keys`: соответствие действий и клавиш.
- `confidence_threshold`: порог уверенности.
- `safe_mode`: блокировка команд при низкой уверенности.
- `human_in_the_loop`: для ручного контроля.
- `only_demo`: принудительно не отправлять клавиши.
- `training_mode` / `inference_mode`: переключатели режимов.


## Если при создании PR ошибка «Бинарные файлы не поддерживаются»
Это нормально для некоторых веб-интерфейсов PR. В этом проекте бинарные артефакты не хранятся в git:
- `models/*.pt`, `models/*.onnx`
- `data/demo_dataset/frames/*.png`

Сгенерируйте их локально перед запуском:
```bash
python train.py --bootstrap-demo --epochs 1
```

Если файлы уже попали в индекс git, удалите их из PR:
```bash
git rm --cached models/*.pt models/*.onnx
git rm --cached data/demo_dataset/frames/*.png
```
После этого закоммитьте изменения и создайте PR снова.

## Ограничения и риски
- Модель обучается на ваших данных и без качественного датасета может ошибаться.
- Захват экрана и управление клавишами зависит от ОС, прав процесса и активного окна.
- В сетевых играх использование подобных инструментов может нарушать пользовательские соглашения/античит.
- Рекомендуется использовать только в одиночных или локальных тестовых сценах.

## Профилирование и оптимизация
- Уменьшайте `image_size` (например, 96x96 → 64x64).
- Снижайте `inference_fps` при слабом CPU.
- Используйте TorchScript/ONNX для ускорения.
- Можно добавить пост-тренировочную квантизацию (int8) при необходимости.
