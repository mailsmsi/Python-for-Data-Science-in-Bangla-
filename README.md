# Python-for-Data-Science-in-Bangla-
Python for Data Science in Bangla (বাংলায় পাইথন ডেটা সায়েন্স)

# Python for Data Science in Bangla (বাংলায় পাইথন ডেটা সায়েন্স)

## প্রাথমিক ধারণা (Basic Concepts)

ডেটা সায়েন্সে পাইথনের ব্যবহার দিন দিন বৃদ্ধি পাচ্ছে। নিচে বাংলায় কিছু মৌলিক ধারণা দেওয়া হলো:

```python
# বাংলায় কমেন্ট (Comment)
# ডেটা সায়েন্সের জন্য প্রয়োজনীয় লাইব্রেরি ইম্পোর্ট করা
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ডেটা লোড করা
data = pd.read_csv('data.csv')

# ডেটার প্রথম ৫টি সারি দেখা
print(data.head())
```

## জনপ্রিয় পাইথন লাইব্রেরি (Popular Python Libraries)

1. **Pandas**: ডেটা ম্যানিপুলেশন এবং বিশ্লেষণের জন্য
   ```python
   # বাংলায় উদাহরণ
   # একটি ডেটাফ্রেম তৈরি
   data = {'নাম': ['রহিম', 'করিম', 'সুমাইয়া'], 'বয়স': [25, 30, 22]}
   df = pd.DataFrame(data)
   print(df)
   ```

2. **NumPy**: সংখ্যাত্মক গণনার জন্য
   ```python
   # অ্যারে তৈরি
   arr = np.array([1, 2, 3, 4, 5])
   print(arr * 2)  # প্রতিটি উপাদানকে ২ দিয়ে গুণ করা
   ```

3. **Matplotlib/Seaborn**: ডেটা ভিজুয়ালাইজেশনের জন্য
   ```python
   # বার চার্ট তৈরি
   names = ['রহিম', 'করিম', 'সুমাইয়া']
   marks = [85, 90, 78]
   
   plt.bar(names, marks)
   plt.title('পরীক্ষার ফলাফল')
   plt.xlabel('ছাত্রের নাম')
   plt.ylabel('প্রাপ্ত নম্বর')
   plt.show()
   ```

4. **Scikit-learn**: মেশিন লার্নিং এর জন্য
   ```python
   from sklearn.linear_model import LinearRegression
   
   # একটি সাধারণ লিনিয়ার রিগ্রেশন মডেল
   model = LinearRegression()
   model.fit(X_train, y_train)  # মডেল ট্রেইন করা
   ```

## বাংলায় রিসোর্স (Resources in Bangla)

1. **YouTube চ্যানেল**:
   - Python Bangla Tutorials
   - Data Science Bangladesh
   - Anisul Islam Python Tutorials

2. **ব্লগ/ওয়েবসাইট**:
   - pythonbangla.com
   - datascienceschool.net/bangla

3. **ফেসবুক গ্রুপ**:
   - Python Programmers Bangladesh
   - Data Science Enthusiasts BD

## ডেটা প্রিপ্রসেসিং উদাহরণ (Data Preprocessing Example)

```python
# বাংলায় কমেন্ট সহ
# মিসিং ভ্যালু হ্যান্ডেল করা
data.fillna(data.mean(), inplace=True)

# ক্যাটাগরিকাল ডেটাকে নিউমেরিকে রূপান্তর
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['লিঙ্গ'] = encoder.fit_transform(data['লিঙ্গ'])

# ডেটা নরমালাইজেশন
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['বয়স', 'আয়']] = scaler.fit_transform(data[['বয়স', 'আয়']])
```

ডেটা সায়েন্সে পাইথন শেখার জন্য নিয়মিত অনুশীলন এবং প্রকৃত প্রজেক্টে কাজ করা গুরুত্বপূর্ণ। বাংলায় অনেক রিসোর্স এখন পাওয়া যায় যা আপনাকে সহজেই শিখতে সাহায্য করবে।
