# تحلیل حساسیت (Sensitivity Analysis) و تست با دیتاست واقعی

## خلاصه

این ماژول‌ها امکان تست کامل با دیتاست واقعی AMPds2 و تحلیل حساسیت پارامترهای کلیدی را فراهم می‌کنند.

---

## فایل‌های جدید

### 1. `sensitivity_analysis.py`
**تحلیل حساسیت پارامترهای کلیدی**

تحلیل حساسیت برای:
- **R (Thermal Resistance)**: مقاومت حرارتی ساختمان
- **C (Thermal Capacitance)**: ظرفیت حرارتی ساختمان
- **w1, w2, w3**: وزن‌های تابع پاداش
- **min_cycle_time**: حداقل زمان چرخه (برای جلوگیری از short-cycling)

**خروجی‌ها**:
- فایل‌های CSV با نتایج
- نمودارهای حساسیت برای هر پارامتر
- گزارش خلاصه

### 2. `test_real_dataset.py`
**تست با دیتاست واقعی AMPds2**

قابلیت‌ها:
- بارگذاری خودکار دیتاست واقعی AMPds2
- پیش‌پردازش و تمیز کردن داده
- تست کامل با Baseline، PI-DRL، و DRL بدون penalty
- تولید جداول با نتایج واقعی

### 3. `run_full_analysis.py`
**اسکریپت کامل برای اجرای همه تحلیل‌ها**

اجرای کامل:
1. تست با دیتاست واقعی
2. تحلیل حساسیت
3. تولید همه خروجی‌ها (figures + tables)

---

## نحوه استفاده

### تحلیل حساسیت

```python
from sensitivity_analysis import SensitivityAnalyzer

# ایجاد analyzer
analyzer = SensitivityAnalyzer()

# تحلیل حساسیت R
results_R = analyzer.analyze_thermal_resistance(
    R_values=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    n_samples=2000
)

# تحلیل حساسیت C
results_C = analyzer.analyze_thermal_capacitance(
    C_values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    n_samples=2000
)

# تحلیل حساسیت وزن‌های پاداش
weight_results = analyzer.analyze_reward_weights(
    w1_values=[0.5, 1.0, 1.5, 2.0],
    w2_values=[5.0, 10.0, 15.0, 20.0],
    w3_values=[0, 2.5, 5.0, 7.5, 10.0],
    n_samples=2000
)

# تحلیل کامل همه پارامترها
all_results = analyzer.comprehensive_analysis(
    save_dir="./sensitivity_results",
    n_samples=2000
)
```

### تست با دیتاست واقعی

```python
from test_real_dataset import test_with_real_data

# تست با دیتاست واقعی
results = test_with_real_data(
    data_path="./data/ampds2/ampds2_data.csv",
    n_samples=5000
)
```

### اجرای کامل

```bash
python3 run_full_analysis.py
```

یا:

```python
from run_full_analysis import main
main()
```

---

## ساختار خروجی‌ها

### تحلیل حساسیت (`./sensitivity_results/`)

```
sensitivity_results/
├── sensitivity_R.csv              # نتایج حساسیت R
├── sensitivity_R.png              # نمودار حساسیت R
├── sensitivity_C.csv              # نتایج حساسیت C
├── sensitivity_C.png              # نمودار حساسیت C
├── sensitivity_w1.csv             # نتایج حساسیت w1
├── sensitivity_w1.png
├── sensitivity_w2.csv             # نتایج حساسیت w2
├── sensitivity_w2.png
├── sensitivity_w3.csv             # نتایج حساسیت w3
├── sensitivity_w3.png
├── sensitivity_cycle_time.csv     # نتایج حساسیت cycle time
├── sensitivity_cycle_time.png
└── sensitivity_summary.txt        # گزارش خلاصه
```

### تست با دیتاست واقعی (`./tables_real_data/`)

```
tables_real_data/
├── table1_hyperparameters.tex
├── table1_hyperparameters.csv
├── table2_performance.tex
├── table2_performance.csv
├── table3_ablation.tex
└── table3_ablation.csv
```

---

## پارامترهای تحلیل حساسیت

### 1. Thermal Resistance (R)
**مقاومت حرارتی ساختمان**

مقادیر پیش‌فرض: `[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]` K/kW

**تأثیر**:
- R کوچکتر → انتقال حرارت بیشتر → نیاز به HVAC بیشتر
- R بزرگتر → عایق بهتر → نیاز به HVAC کمتر

### 2. Thermal Capacitance (C)
**ظرفیت حرارتی ساختمان**

مقادیر پیش‌فرض: `[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]` kWh/K

**تأثیر**:
- C کوچکتر → تغییر سریع دما → نیاز به کنترل بیشتر
- C بزرگتر → تغییر آهسته دما → کنترل راحت‌تر

### 3. Reward Weights

#### w1 (Cost Weight)
مقادیر پیش‌فرض: `[0.5, 1.0, 1.5, 2.0]`

**تأثیر**: وزن بیشتر → اولویت بیشتر به کاهش هزینه

#### w2 (Discomfort Weight)
مقادیر پیش‌فرض: `[5.0, 10.0, 15.0, 20.0]`

**تأثیر**: وزن بیشتر → اولویت بیشتر به راحتی حرارتی

#### w3 (Cycling Penalty Weight)
مقادیر پیش‌فرض: `[0, 2.5, 5.0, 7.5, 10.0]`

**تأثیر**: 
- w3=0 → بدون penalty → short-cycling زیاد
- w3>0 → penalty بیشتر → جلوگیری از short-cycling

### 4. Minimum Cycle Time
**حداقل زمان چرخه**

مقادیر پیش‌فرض: `[5, 10, 15, 20, 25, 30]` minutes

**تأثیر**:
- زمان کمتر → انعطاف بیشتر اما short-cycling بیشتر
- زمان بیشتر → محافظت بیشتر از سخت‌افزار اما انعطاف کمتر

---

## استفاده از دیتاست واقعی AMPds2

### مرحله 1: دانلود دیتاست

1. ثبت‌نام در: https://ampds.org/
2. دانلود دیتاست AMPds2
3. استخراج فایل‌های CSV

### مرحله 2: آماده‌سازی داده

فایل CSV باید این ستون‌ها را داشته باشد:
- `timestamp`: زمان (DateTime)
- `WHE`: مصرف انرژی آبگرمکن
- `HPE`: مصرف انرژی پمپ حرارتی
- `FRE`: مصرف انرژی یخچال
- `Outdoor_Temp`: دمای بیرون
- `Solar_Rad`: تابش خورشید
- `Price`: قیمت برق (اختیاری - اگر نباشد تولید می‌شود)

### مرحله 3: قرار دادن در پوشه

```
./data/ampds2/
└── your_ampds2_data.csv
```

### مرحله 4: اجرای تست

```python
from test_real_dataset import test_with_real_data

results = test_with_real_data(
    data_path="./data/ampds2/your_ampds2_data.csv",
    n_samples=5000
)
```

---

## تفسیر نتایج تحلیل حساسیت

### مثال: تحلیل R

```python
results_R = analyzer.analyze_thermal_resistance()

# نتایج:
#   value  cost  discomfort  cycles
#   0.02   12.5   35000       150
#   0.05   10.4   31000       120  <- base
#   0.08    8.2   28000        95
```

**تفسیر**:
- R بزرگتر → هزینه کمتر (عایق بهتر)
- اما ممکن است راحتی کمتر (کنترل سخت‌تر)

### مثال: تحلیل w3 (Cycling Penalty)

```python
results_w3 = analyzer.analyze_parameter_sensitivity('w3', [0, 2.5, 5.0, 7.5, 10.0])

# نتایج:
#   value  cost  cycles  violations
#   0      9.5   200     150        <- بدون penalty
#   5.0   10.4   120      0         <- base
#   10.0  11.2   100      0         <- penalty زیاد
```

**تفسیر**:
- w3=0 → هزینه کمتر اما violations زیاد (تخریب سخت‌افزار)
- w3>0 → هزینه کمی بیشتر اما محافظت از سخت‌افزار

---

## نکات مهم

1. **زمان اجرا**: تحلیل حساسیت کامل ممکن است چند ساعت طول بکشد
   - برای تست سریع: `n_samples=500`
   - برای نتایج دقیق: `n_samples=2000-5000`

2. **دیتاست واقعی**: 
   - اگر دیتاست واقعی موجود نباشد، از دیتای سنتتیک استفاده می‌شود
   - برای مقاله باید از دیتاست واقعی استفاده کنید

3. **پارامترهای قابل تنظیم**:
   - همه مقادیر پیش‌فرض قابل تغییر هستند
   - می‌توانید پارامترهای خاص را تحلیل کنید

---

## مثال کامل

```python
# 1. تحلیل حساسیت
from sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer()
results = analyzer.comprehensive_analysis(n_samples=2000)

# 2. تست با دیتاست واقعی
from test_real_dataset import test_with_real_data

real_results = test_with_real_data(n_samples=5000)

# 3. مقایسه نتایج
print("Synthetic data results:", results)
print("Real data results:", real_results)
```

---

## خروجی‌های برای مقاله

تحلیل حساسیت می‌تواند به عنوان:
1. **جدول حساسیت** در مقاله
2. **نمودارهای حساسیت** (Figure)
3. **بحث درباره پارامترها** در بخش Results

استفاده شود.

---

**نکته**: برای مقاله Applied Energy، تحلیل حساسیت نشان می‌دهد که نتایج شما robust هستند و به پارامترهای خاص وابسته نیستند.
