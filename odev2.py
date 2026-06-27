import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Hataları ezecek yeni aracımız

# 1. Veriyi Okuma (Dosyada başlık olduğu için header=None sildik!)
dosya_yolu = r"C:\Users\Emirhan Kunt\Desktop\Programlama\Python Klasörü\data\housevotes84.csv"
data = pd.read_csv(dosya_yolu)

sutunlar = [
    'parti', 'engelli_bebekler', 'su_projesi', 'butce_tasarisi', 'doktor_ucreti', 
    'el_salvador', 'okulda_din', 'anti_uydu', 'nikaragua', 'mx_fuzesi', 
    'gocmenlik', 'sentetik_yakit', 'egitim_butcesi', 'cevre_davasi', 
    'suc_yasasi', 'gumruksuz_ihracat', 'guney_afrika'
]

data.columns = sutunlar
# 2. Bağımlı (y) ve Bağımsız (x) Değişkenleri Ayırma
y_ham = data.iloc[:, 0]  # İlk sütun (Partiler)
x_ham = data.iloc[:, 1:] # Geri kalanlar (Oylar)

# 3. Kusursuz Encoding İşlemi
# y (Partiler) için LabelEncoder kullanıyoruz. PyArrow tip hatalarına takılmadan sayılara çevirir.
le = LabelEncoder()
y = le.fit_transform(y_ham).reshape(-1, 1)

# x (Oylar) için harfleri sayılara çevirip, sütunların sayısal (int) olmasını garanti altına alıyoruz.
x = x_ham.replace({'y': 1, 'n': -1, '?': 0}).astype(int)

# 4. Veriyi Parçalama
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)

# 2. 9 MODELİ TANIMLAMA (Notlarındaki modellerin birebir aynısı)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
modeller = {
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(kernel="rbf"),
    "Yapay Sinir Ağları (MLP)": MLPClassifier(max_iter=1000, random_state=34),
    "Karar Ağacı (CART)": DecisionTreeClassifier(random_state=34),
    "Random Forest": RandomForestClassifier(random_state=34),
    "Gradient Boosting": GradientBoostingClassifier(random_state=34),
    "Lojistik Regresyon": LogisticRegression(solver="liblinear"),
    "Naive Bayes": GaussianNB(),
    "K-En Yakın Komşu (KNN)": KNeighborsClassifier()
}

# 3. YARIŞMA DÖNGÜSÜ
sonuclar = []

for isim, model in modeller.items():
    # Modeli eğit (.fit)
    model.fit(X_train, y_train)
    # Test seti ile tahmin yap (.predict)
    tahminler = model.predict(X_test)
    
    # Başarı metriklerini hesapla
    acc = accuracy_score(y_test, tahminler)
    f1 = f1_score(y_test, tahminler)
    
    # Sonuç listesine ekle
    sonuclar.append({"Model": isim, "Doğruluk Oranı": round(acc, 4), "F1 Skoru": round(f1, 4)})

# 4. SONUÇLARI SIRALAMA VE GÖSTERME
sonuclar_df = pd.DataFrame(sonuclar)
sonuclar_df = sonuclar_df.sort_values(by="Doğruluk Oranı", ascending=False).reset_index(drop=True)

print("--- 1984 ABD KONGRESİ KUTUPLAŞMA TAHMİNİ SONUÇLARI ---")
print(sonuclar_df)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Eğitilmiş Random Forest Modelini Yarışmadan Çekiyoruz
rf_model = modeller["Random Forest"]

# 2. Algoritmanın yasalara verdiği önem derecelerini alıyoruz (Matematiksel ağırlıklar)
onem_dereceleri = rf_model.feature_importances_

# 3. Yasa isimleriyle bu oranları eşleştirip bir Tablo (DataFrame) yapıyoruz
# Not: Hedef değişken (Parti) 0. indekste olduğu için, 1. indeksten sonrasını (yasaları) alıyoruz.
yasa_isimleri = data.columns[1:]

onem_df = pd.DataFrame({
    "Yasa Tasarısı": yasa_isimleri, 
    "Etki Oranı": onem_dereceleri
})

# Tabloyu en yüksek etkiden en düşüğe doğru sıralıyoruz
onem_df = onem_df.sort_values(by="Etki Oranı", ascending=False)

# 4. Şık Bir Grafik Çizdiriyoruz
plt.figure(figsize=(10, 6))
sns.barplot(x="Etki Oranı", y="Yasa Tasarısı", data=onem_df, palette="magma")

plt.title("1984 ABD Kongresi: Kutuplaşmayı Yaratan En Kritik Yasa Tasarıları", fontsize=14, fontweight="bold")
plt.xlabel("Model İçindeki Etki Oranı (Feature Importance)", fontsize=12)
plt.ylabel("Yasa Tasarıları", fontsize=12)
plt.tight_layout()
plt.show()

# İstersen metin olarak da en önemli 3 yasayı yazdıralım:
print("\n--- MECLİSİ BÖLEN EN ÖNEMLİ 3 YASA ---")
print(onem_df.head(3).to_string(index=False))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Şampiyon Modeli (Lojistik Regresyon) Çekiyoruz
lr_model = modeller["Lojistik Regresyon"]

# 2. Modelin yasalara atadığı Ağırlıkları (Katsayıları) alıyoruz
# [0] yazmamızın sebebi katsayıların liste içinde liste olarak gelmesidir
katsayilar = lr_model.coef_[0]

# 3. Yasa isimleriyle katsayıları eşleştiriyoruz
yasa_isimleri = data.columns[1:]

katsayi_df = pd.DataFrame({
    "Yasa Tasarısı": yasa_isimleri, 
    "Katsayı (Ağırlık)": katsayilar
})

# Katsayıları büyüklüklerine göre (negatiften pozitife) sıralıyoruz
katsayi_df = katsayi_df.sort_values(by="Katsayı (Ağırlık)")

# 4. Yönü Belli Olan Harika Bir Grafik Çizdiriyoruz
plt.figure(figsize=(10, 6))
# Katsayı 0'dan büyükse mavi (Demokrat), küçükse kırmızı (Cumhuriyetçi) tonlar kullansın
renkler = ['red' if x < 0 else 'blue' for x in katsayi_df["Katsayı (Ağırlık)"]]

sns.barplot(x="Katsayı (Ağırlık)", y="Yasa Tasarısı", data=katsayi_df, palette=renkler)

plt.title("Lojistik Regresyon: Yasaların Kutuplaşma Yönü ve Ağırlığı", fontsize=14, fontweight="bold")
plt.xlabel("Model Katsayısı (Negatif: Sınıf 0'a çeker, Pozitif: Sınıf 1'e çeker)", fontsize=12)
plt.ylabel("Yasa Tasarıları", fontsize=12)
plt.axvline(0, color='black', linewidth=1) # Sıfır noktasına çizgi çekiyoruz
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Diğer Şampiyon Modeli (Linear SVM) Çekiyoruz
svm_model = modeller["Linear SVM"]

# 2. Modelin yasalara atadığı Ağırlıkları (Katsayıları) alıyoruz
katsayilar_svm = svm_model.coef_[0]

# 3. Yasa isimleriyle katsayıları eşleştiriyoruz
yasa_isimleri = data.columns[1:]

katsayi_svm_df = pd.DataFrame({
    "Yasa Tasarısı": yasa_isimleri, 
    "Katsayı (Ağırlık)": katsayilar_svm
})

# Katsayıları büyüklüklerine göre sıralıyoruz
katsayi_svm_df = katsayi_svm_df.sort_values(by="Katsayı (Ağırlık)")

# 4. Linear SVM Grafiğini Çizdiriyoruz
plt.figure(figsize=(10, 6))
renkler_svm = ['red' if x < 0 else 'blue' for x in katsayi_svm_df["Katsayı (Ağırlık)"]]

sns.barplot(x="Katsayı (Ağırlık)", y="Yasa Tasarısı", data=katsayi_svm_df, palette=renkler_svm)

plt.title("Linear SVM: Sınır Çizgisini Belirleyen Yasalar", fontsize=14, fontweight="bold")
plt.xlabel("Model Katsayısı (Margin/Sınır Ağırlığı)", fontsize=12)
plt.ylabel("Yasa Tasarıları", fontsize=12)
plt.axvline(0, color='black', linewidth=1) 
plt.tight_layout()
plt.show()

import pandas as pd

# 1. Önce harfleri kesin olarak doğru etiketlerle değiştiriyoruz
el_salvador_temiz = data['el_salvador'].map({'n': 'Hayır (-1)', '?': 'Çekimser (0)', 'y': 'Evet (1)'})

# 2. Çapraz Tabloyu (Crosstab) oluşturuyoruz
el_salvador_tablosu = pd.crosstab(index=data['parti'], columns=el_salvador_temiz)

# 3. Tablodaki sütunları mantıklı bir sıraya diziyoruz
el_salvador_tablosu = el_salvador_tablosu[['Hayır (-1)', 'Çekimser (0)', 'Evet (1)']]

# 4. Tablo başlıklarını temizliyoruz
el_salvador_tablosu.index.name = "PARTİLER"
el_salvador_tablosu.columns.name = "EL SALVADOR OYU"

print("--- EL SALVADOR YASASI İÇİN GERÇEK İSYAN TABLOSU ---")
print(el_salvador_tablosu)

import pandas as pd

# 1. Doktor ücretleri için harfleri kesin etiketlerle değiştiriyoruz
doktor_temiz = data['doktor_ucreti'].map({'n': 'Hayır (-1)', '?': 'Çekimser (0)', 'y': 'Evet (1)'})

# 2. Çapraz Tabloyu oluşturuyoruz
doktor_tablosu = pd.crosstab(index=data['parti'], columns=doktor_temiz)

# 3. Tablodaki sütunları mantıklı bir sıraya diziyoruz
doktor_tablosu = doktor_tablosu[['Hayır (-1)', 'Çekimser (0)', 'Evet (1)']]

# 4. Tablo başlıklarını temizliyoruz
doktor_tablosu.index.name = "PARTİLER"
doktor_tablosu.columns.name = "DOKTOR ÜCRETLERİ OYU"

print("--- DOKTOR ÜCRETLERİ YASASI İÇİN PARTİ DİSİPLİNİ TABLOSU ---")
print(doktor_tablosu)