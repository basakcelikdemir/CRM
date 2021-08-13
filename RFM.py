######################################GOREV 1############################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#1)Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz
df_ = pd.read_excel("Dersler/HAFTA3/Ders Notları/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head(10)
df.shape

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.


# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


#2)Veri setinin betimsel istatistiklerini inceleyiniz
df.info()
df.head()
df.describe()

#3) Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?

df.shape
df.isnull().sum()

#4)Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.

df.dropna(inplace=True)
df.shape

#5) eşsiz ürün sayısı
df["StockCode"].nunique()


#6)Hangi üründen kaçar tane vardır?

df["StockCode"].value_counts().head(10)

#6) En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız.

df["Quantity"].sort_values(ascending=False).head()

#7)8. Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.

df = df[~df["Invoice"].str.contains("C", na=False)]
df.shape

#9)Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

######################################GOREV 2############################
"""
RFM metriklerinin hesaplanması:

Recency, Frequency ve Monetary tanımlarını yapınız.
> Müşteri özelinde Recency, Frequency ve Monetary metriklerini groupby, agg ve lambda ile
hesaplayınız.
> Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
> Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz

Not 1: recency değeri için bugünün tarihini (2011, 12, 11) olarak kabul ediniz.
Not 2: rfm dataframe’ini oluşturduktan sonra veri setini "monetary>0" olacak şekilde filtreleyiniz.
"""
# Recency (yenilik): Müşterinin son satın almasından bugüne kadar geçen süre
# Frequency (Sıklık): Toplam satın alma sayısı.
# Monetary (Parasal Değer): Müşterinin yaptığı toplam harcama.

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()
rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]

######################################GOREV 3############################
"""
RFM skorlarının oluşturulması ve tek bir değişkene çevrilmesi

> Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
> Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
> Oluşan 3 farklı değişkenin değerini tek bir değişken olarak ifade ediniz ve RFM_SCORE olarak kaydediniz. 

Örneğin;
Ayrı ayrı değişkenlerde sırasıyla 5, 2, 1 olan recency_score, frequency_score ve monetary_score skorlarını
RFM_SCORE değişkeni isimlendirmesi ile 521 olarak oluşturunuz

"""
#Müşterinin son satın almasından bugüne kadar geçen süre.farkı en büyük olan 1'i en küçük olan 5'i ifade eder
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# Alışveriş sıklığı skoru. Burada 1 en az sıklığı, 5 en fazla sıklığı temsil eder.
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Bize bıraktığı para tutarı. Burada 1 en az parayı, 5 en fazla parayı temsil eder.
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))
rfm.head()

######################################GOREV 4############################
"""
RFM skorlarının segment olarak tanımlanması

> Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlamaları
yapınız.

> seg_map yardımı ile skorları segmentlere çeviriniz.

"""

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]':'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True) # birleştirilen skorlar seg_map ile değiştirildi
rfm.head(15)

######################################GOREV 5############################
"""
Aksiyon zamanı!

> > Önemli bulduğunuz 3 segmenti seçiniz. Bu üç segmenti;
>Hem aksiyon kararları açısından,
>Hem de segmentlerin yapısı açısından (ortalama RFM değerleri) yorumlayınız.
"""

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

"""
Risk,can't loose ve need attention'a yatırımlar yapıp çeşitli kampanyalarla
(ilgili kampanyada ki asıl amaç alışveriş alışkanlığı için) aksiyonlar alınabilir.
bu üç segment recency bakımından today date'e  en uzaklardan 
oldukları için bir kampanyayla  kapsanmaya çalışılabilir.
"""

# >>"Loyal Customers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.

new_df = pd.DataFrame()
new_df["loyal_customers"] = rfm[rfm["segment"] == "loyal_customers"].index
new_df.head()
new_df.to_csv("new_customers.csv")  # df'i kaydet
