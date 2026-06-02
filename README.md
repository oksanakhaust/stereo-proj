# Stereo-Proj

Генератор стандартных стереографических проекций для кубической кристаллографической системы.

## Возможности

- Корректные стереографические проекции по формуле **r = R · tan(ρ/2)**
- Примитивные плоскости (hkl): автоматически исключаются кратные направления
- Сетка Вульфа: концентрические окружности + радиальные линии
- Верхняя/нижняя/обе полусферы (закрашенные и контурные кружки)
- Подписи (hkl) с барным обозначением отрицательных индексов
- Экспорт: **PNG, JPEG, PDF** (300 dpi)
- Встраивается в любую Python-платформу: Jupyter, Flask, FastAPI, Streamlit

---

## Быстрый старт

### 1. Установка

```bash
pip install -e ".[dev]"
```

### 2. Веб-интерфейс (Streamlit)

```bash
streamlit run app.py
```

Откройте `http://localhost:8501` в браузере.

### 3. Python API

```python
from stereo_proj import generate

renderer = generate(
    center=(0, 0, 1),     # центр проекции [HKL]
    max_sum_sq=9,          # h²+k²+l² ≤ 9
    radius=100,
    hemisphere="both",     # 'upper' | 'lower' | 'both'
    show_grid=True,
    show_labels=True,
)

renderer.export("cubic_001.png", "png")      # сохранить файл
fig  = renderer.get_figure()                 # matplotlib Figure
buf  = renderer.get_bytes("jpeg")            # BytesIO для web
```

---

## Параметры

| Параметр | По умолчанию | Ограничение |
|----------|-------------|-------------|
| `max_sum_sq` | 9 | ≤ 25 |
| Радиус сетки | 100 | 50–300 |
| Индексы h, k, l | — | −10 … 10 |
| Макс. полюсов | — | 1000 |

---

## Математика

**Проекция:**
Для полюса `[hkl]` с угловым расстоянием ρ от центра `[HKL]`:

```
r = R · tan(ρ/2)
x = r · cos(φ),  y = r · sin(φ)
```

**Азимутальный угол φ** вычисляется через ортонормированный базис
`{e₁, e₂}` в плоскости, перпендикулярной центральному полюсу.

**Задняя полусфера** (ρ > 90°) проецируется как антиподальный полюс
с `r = R·tan((π−ρ)/2)` и отображается пустым кружком.

---

## Структура проекта

```
stereo_proj/
├── src/stereo_proj/
│   ├── crystal/
│   │   ├── base.py        # абстрактный CrystalSystem + константы
│   │   └── cubic.py       # CubicSystem: генерация hkl, углы
│   ├── projection.py      # StereographicProjection: математика
│   ├── renderer.py        # StereogramRenderer: matplotlib + экспорт
│   └── __init__.py        # публичный API + функция generate()
├── app.py                 # Streamlit веб-интерфейс
├── tests/
│   ├── test_cubic.py
│   └── test_projection.py
└── examples/
    ├── basic_001.py
    └── custom_center.py
```

---

## Тесты

```bash
pytest tests/ -v
```

---

## Планируемые расширения

- Гексагональная система (добавить `HexagonalSystem` в `crystal/`)
- Тетрагональная система
- Зональные круги (great circles)
- Интерактивные подсказки при наведении (Plotly / Bokeh)
