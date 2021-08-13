
##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

pip install lifetimes
pip install mysql-connector-python
pip install sqlalchemy
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit





df = retail_mysql_df.copy()

############################################Görev 1 #################################################
"""
6 aylık CLTV Prediction
§ 2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
§ Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.

DİKKAT!
6 aylık expected number of transaction değil cltv prediction yapılmasını
beklenmektedir.
Yani direkt BGNBD & GAMMA GAMMA modellerini kurarak devam ediniz ve
cltv prediction için ay bölümüne 6 giriniz
"""


df.describe().T
df.dropna(inplace=True) #eksik gözlemler silindir
df = df[~df["Invoice"].str.contains("C", na=False)] #iade siparişler dışındakiler seçilip atandı
df = df[df["Quantity"] > 0] #hiç ödeme yapmamışlar çıkarıldı

replace_with_thresholds(df, "Quantity") #aykırı değerler baskılandı
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"] # Ürün adedi * fiyat

df=df[df["Country"]=="United Kingdom"] #UK müşteriler seçildi sadece
df.shape
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11) # analizi yapacağımız gün belirtildi


# Lifetime Veri Yapısının Hazırlanması


# recency: Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["monetary"] > 0]

# BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency'nin 1'den büyük olması gerekmektedir.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]



#  BG-NBD Modelinin Kurulması


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["expected_purc_1_week"]=bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])


cltv_df.sort_values(by="expected_purc_1_week", ascending=False).head(15)

cltv_df["expected_purc_1_month"]=bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])




# Tahmin Sonuçlarının Değerlendirilmesi


plot_period_transactions(bgf)
plt.show()


#  GAMMA-GAMMA Modelinin Kurulması


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])



cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)



# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.



cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)


# CLTV'nin Standartlaştırılması
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])



# Sıralayalım:
cltv_final.sort_values(by="scaled_clv", ascending=False).head()

# Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.
cltv_final.sort_values(by="scaled_clv", ascending=False).head(15)

"""
2486 ve 587 index nolu customerleri karşılaştırdığımda
587 recency bakımından ve monetary bakımından 2468 noludan değerleri daha iyidir. 
fakat demekki burada frequency ve müşteri yaşı  etkili bbir rol oynuyor
"""

#################################################Görev 2 ############################################

"""
Farklı zaman periyotlarından oluşan CLTV analizi
§ 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
§ 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
§ Fark var mı? Varsa sizce neden olabilir?
DİKKAT!
Sıfırdan model kurulmasına gerek yoktur.
Önceki soruda oluşturulan model üzerinden ilerlenebilir.
"""

def create_cltv_p(dataframe, month=3):

    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]

    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")

    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)
    cltv_df = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                    lambda date: (today_date - date.min()).days],
                                                    'Invoice': lambda num: num.nunique(),
                                                    'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[cltv_df["monetary"] > 0]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_final[["clv"]])
    cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

    cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

create_cltv_p(df,month=1).sort_values(by="clv",ascending=False).head(10)

create_cltv_p(df,month=12).sort_values(by="clv",ascending=False).head(10)


"""
1.ayda cltv değeri fazla olanlar 12.ayda da en fazladırlar.2 müşteri hariç
Bunlara segment bakımından bakacak olursam demek ki A segmenti istikrarlı devam eden bir tayfa:)
"""

#################################################Görev 3 ############################################
"""
Segmentasyon ve Aksiyon Önerileri

§ 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba
(segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

§ 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon
önerilerinde bulununuz

"""


cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
""
cltv_final.head()

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

"""
B segmenti ve C segmenti belli bir düzey alışveriş alışkanlıkları olduğu için bu iki grup üzerinde kampanyalar planlardım
amacım B segmentini A segmenti düzeyine çıkarmak,C'yi B. 

İki segmente de özel mailler,aileden biriymiş gibi hissettiren mailler yollayıp sadece onlara özel 
{diyelim ki bu iki segment toplamı 200 kişi} 100 adet pahalı bir ürün verecek çekiliş ayarlardım,
10 alışverişe bir çekiliş hakkı ve bütün alışverişlerine kargo bedava koyardım.
çekiliş sayfasında çekilişe başvuran sayısını gösteren tablo  koyardım.Böylelikle kazanma olasılıklarını görünür.
ve  %50 kazanan müşterilere hediyelerini yollardım
tam olarak kar zarar ne kadar olur marka için bilemiyorum fakat böylelikle güven duygusu artar,alışveriş alışkanlığı kazanılır ve diğer parametreler artardı

"""

