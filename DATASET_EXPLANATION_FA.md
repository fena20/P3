# توضیح دقیق دیتاست استفاده شده

## خلاصه

**دیتاست واقعی استفاده نشده است!** 

از **دیتای سنتتیک (Synthetic/Mock Data)** استفاده شده که الگوهای دیتاست **AMPds2** را شبیه‌سازی می‌کند.

---

## جزئیات دقیق

### 1. دیتاست واقعی: AMPds2

**AMPds2** یک دیتاست معروف برای مصرف انرژی ساختمان‌های مسکونی است:
- **نام کامل**: "AMPds2: The Almanac of Minutely Power dataset Series 2"
- **نویسندگان**: Makonin et al.
- **رزولوشن**: 1 دقیقه (1-minute resolution)
- **محتوای اصلی**: 
  - WHE (Water Heater Energy)
  - HPE (Heat Pump Energy) 
  - FRE (Fridge Energy)
  - دما و سایر پارامترها

### 2. دیتای سنتتیک تولید شده

به جای استفاده از دیتاست واقعی، کد **دیتای سنتتیک** تولید می‌کند که:

#### ویژگی‌های دیتای سنتتیک:

1. **رزولوشن**: 1 دقیقه (مثل AMPds2)
2. **مدت زمان**: 365 روز (قابل تنظیم)
3. **ستون‌ها**:
   - `WHE`: مصرف انرژی آبگرمکن (با الگوی تصادفی)
   - `HPE`: مصرف انرژی پمپ حرارتی (کنترل شده توسط agent)
   - `FRE`: مصرف انرژی یخچال (تقریباً ثابت)
   - `Outdoor_Temp`: دمای بیرون (الگوی سینوسی روزانه و فصلی)
   - `Solar_Rad`: تابش خورشید (صفر در شب، پیک در ظهر)
   - `Price`: قیمت برق (Time-of-Use: بالاتر در ساعات پیک 17-20)

#### نحوه تولید دیتای سنتتیک:

```python
# در فایل data_loader.py

def generate_synthetic_ampds2_data():
    # دمای بیرون: الگوی سینوسی فصلی + روزانه + نویز
    seasonal_temp = 15 + 10 * sin(2π * day/365)
    daily_temp = 5 * sin(2π * hour/24)
    outdoor_temp = seasonal_temp + daily_temp + noise
    
    # تابش خورشید: صفر در شب، پیک در ظهر
    solar_rad = 800 * sin(π * (hour - 6) / 12)  # 6 صبح تا 6 عصر
    
    # قیمت برق: پایه 0.10 $/kWh، 1.5 برابر در ساعات پیک
    price = 0.10 * (1.5 if 17 <= hour < 20 else 1.0)
    
    # مصرف انرژی: الگوهای تصادفی با توزیع نمایی
    WHE = exponential(0.5) + spikes
    HPE = normal(2.5) if outdoor_temp < 18 else normal(0.1)
    FRE = normal(0.15)
```

---

## چرا دیتای سنتتیک؟

### مزایا:
1. ✅ **فوری قابل استفاده**: نیاز به دانلود دیتاست واقعی نیست
2. ✅ **قابل تکرار**: نتایج یکسان با seed یکسان
3. ✅ **قابل تنظیم**: پارامترها را می‌توان تغییر داد
4. ✅ **الگوهای واقعی**: بر اساس ویژگی‌های AMPds2 طراحی شده

### معایب:
1. ⚠️ **دیتای واقعی نیست**: برای مقاله باید از دیتاست واقعی استفاده کنید
2. ⚠️ **الگوهای ساده**: پیچیدگی‌های دنیای واقعی را ندارد

---

## چگونه از دیتاست واقعی استفاده کنیم؟

### روش 1: دانلود AMPds2

1. دانلود دیتاست از:
   - [AMPds2 Dataset](https://ampds.org/)
   - یا از منابع آکادمیک

2. تبدیل به فرمت CSV با ستون‌های:
   ```csv
   timestamp,WHE,HPE,FRE,Outdoor_Temp,Solar_Rad,Price
   2012-04-01 00:00:00,0.5,2.3,0.15,12.5,0,0.10
   ...
   ```

3. استفاده در کد:
   ```python
   from data_loader import load_ampds2_data
   
   # استفاده از دیتاست واقعی
   data = load_ampds2_data(filepath="path/to/ampds2_real_data.csv")
   ```

### روش 2: استفاده از دیتاست‌های دیگر

می‌توانید دیتاست‌های مشابه را استفاده کنید:
- **REDD**: Residential Energy Disaggregation Dataset
- **UK-DALE**: UK Domestic Appliance-Level Electricity
- **ECO**: Electricity Consumption and Occupancy

فقط باید فرمت CSV را مطابق با ستون‌های مورد نیاز تنظیم کنید.

---

## فایل‌های مرتبط

### `data_loader.py`
- تابع `generate_synthetic_ampds2_data()`: تولید دیتای سنتتیک
- تابع `load_ampds2_data()`: بارگذاری دیتاست واقعی یا سنتتیک

### نحوه کار:
```python
# اگر فایل وجود نداشته باشد، دیتای سنتتیک تولید می‌شود
env = SmartHomeEnv(data_path=None)  # دیتای سنتتیک

# اگر فایل وجود داشته باشد، از دیتاست واقعی استفاده می‌شود
env = SmartHomeEnv(data_path="ampds2_data.csv")  # دیتاست واقعی
```

---

## برای مقاله Applied Energy

### توصیه:

1. **برای تست و توسعه**: استفاده از دیتای سنتتیک (فعلی) ✅
2. **برای مقاله**: استفاده از دیتاست واقعی AMPds2 ⚠️

### در مقاله باید بنویسید:

```
"We use the AMPds2 dataset (Makonin et al., 2016) with 1-minute resolution 
for residential building energy data. The dataset includes water heater 
energy (WHE), heat pump energy (HPE), fridge energy (FRE), outdoor temperature, 
and solar radiation measurements."
```

### اگر از دیتای سنتتیک استفاده می‌کنید:

```
"We generate synthetic data mimicking AMPds2 patterns for initial testing. 
The synthetic data includes realistic daily and seasonal temperature variations, 
solar radiation patterns, and time-of-use electricity pricing. For final 
results, we use the actual AMPds2 dataset."
```

---

## خلاصه

| مورد | وضعیت فعلی | برای مقاله |
|------|------------|------------|
| **دیتاست** | سنتتیک (Synthetic) | واقعی (AMPds2) |
| **منبع** | تولید شده در کد | دانلود از ampds.org |
| **رزولوشن** | 1 دقیقه ✅ | 1 دقیقه ✅ |
| **ستون‌ها** | WHE, HPE, FRE, Temp, Solar, Price ✅ | مشابه ✅ |
| **قابل استفاده** | برای تست ✅ | برای مقاله ⚠️ |

---

## مراجع

1. **AMPds2 Dataset**: 
   - Makonin, S., Popowich, F., Bartram, L., Gill, B., & Bajić, I. V. (2016). 
   - "AMPds2: The Almanac of Minutely Power dataset Series 2"
   - Available at: https://ampds.org/

2. **کد فعلی**:
   - فایل `data_loader.py` خط 11-90
   - تابع `generate_synthetic_ampds2_data()`

---

**نتیجه‌گیری**: در حال حاضر از دیتای سنتتیک استفاده شده که الگوهای AMPds2 را شبیه‌سازی می‌کند. برای مقاله نهایی، باید از دیتاست واقعی AMPds2 استفاده کنید.
