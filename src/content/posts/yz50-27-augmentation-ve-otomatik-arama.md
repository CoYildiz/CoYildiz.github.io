---
title: "27 · Data Augmentation ve Otomatik Arama"
published: 2026-10-01
description: "Modeli değil veriyi büyütmek: hangi dönüşüm geçerli, sızıntı nerede başlıyor, ve grid/random/Bayesian arama."
tags:
  - YZ50
  - Data Augmentation
  - Hiperparametre
category: ML
draft: false
lang: tr
---

> [!WARNING]
> **Bu notlar nasıl yazıldı**
>
> Bu notlar **yapay zeka ile tartışılarak**, çeşitli kaynaklardan derlenip düzenlenerek
> yazıldı. **Hata veya eksik içerebilir** — nitekim sonradan yapılan bir doğruluk
> taramasında yedi yanlış iddia bulunup düzeltildi. Kritik bir şeyi buradan alıp
> kullanmadan önce birincil kaynaktan doğrula.

**Bu not neden var:** Overfitting'e karşı iki yol var. [25](/posts/yz50-25-regularization-l1-l2/) ve
[26](/posts/yz50-26-hiperparametre-ve-early-stopping/) **modeli** kısıtlamayı anlatıyordu (ceza terimi,
erken durdurma). Bu not diğer yolu anlatıyor: **veriyi çoğaltmak**. Ardından, hiperparametre
aramasını elle yapmayı bırakıp otomatikleştirmeyi.

---

## 1. Data augmentation nedir

Overfitting'in en temiz çözümü daha çok veridir. Ama veri toplamak ve etiketlemek pahalı, çoğu
zaman imkânsız. **Data augmentation**, elindeki veriye gerçek dünyada olabilecek değişiklikleri
uygulayarak yeni örnekler üretmek demek.

Fikir şuradan geliyor: bir '5' rakamını 15 derece döndürürsen, **senin için hâlâ 5'tir**. Ama ağ
piksellere bakıyor ve piksel düzeyinde o tamamen başka bir girdi.

![Aynı rakam, altı dönüşüm](/yz50/augmentation-rakamlar.png)

Altı görüntünün altındaki "piksel farkı" sayıları bunu gösteriyor: insan gözü için hepsi aynı
rakam, ağ için altı farklı girdi. Yani bedava veri.

Nielsen'in MNIST ölçümü: 800 gizli nöronlu bir ağ standart 50.000 görüntüyle **%98.4**, aynı ağ
döndürme/kaydırma/esnek bozulmayla genişletilmiş veriyle **%99.3** alıyor.

### Neden işe yarıyor

Ağ, bir rakamı tanımak için iki yoldan birini seçebilir:

1. Eğitimdeki o **tek görüntünün** piksel desenini ezberlemek
2. Rakamı rakam yapan, döndürülse de kaydırılsa da **değişmeyen** özellikleri öğrenmek

Augmentation birinci yolu kapatıyor. Aynı rakamı on farklı biçimde gördüğünde, hepsinde ortak
olan şeyi bulmak zorunda kalıyor. Yani modele **"bu dönüşümler önemsiz"** diye bir bilgi
öğretmiş oluyorsun — ve bu bilgi veriden değil, senin alan bilginden geliyor.

### Güçlü ve zayıf yanları

| | |
|---|---|
| **Güçlü** | Genelde model kurcalamaktan daha çok kazandırıyor; veri toplamanın bedava alternatifi |
| **Güçlü** | Alan bilgini modele aktarmanın doğrudan yolu — "döndürme önemsiz" bilgisini mimariye gömmek zor, veriye gömmek kolay |
| **Zayıf** | **Dönüşümün geçerli olması gerekiyor.** Rakamı 180° döndürürsen 6 ile 9 karışır — etiketi bozarsın |
| **Zayıf** | Gerçek yeni bilgi eklemiyor; aynı örneklerin varyasyonları, bağımsız yeni örnek değil |
| **Zayıf** | Eğitim süresini doğrudan çarpıyor (3 kat çoğaltma = 3 kat süre) |
| **Zayıf** | Yanlış sırada yapılırsa **test skorunu şişiriyor** — bölüm 3 |

İkinci zayıf madde en kritiği: **hangi dönüşümün geçerli olduğu veriye bağlı ve bunu sen
bilmek zorundasın.** Model bilmiyor.

## 2. Görüntü dışında — senin kullanacağın hali

| Alan | Geçerli dönüşümler | Dikkat |
|---|---|---|
| Görüntü | döndürme, kaydırma, eğme, esnek bozulma | 6↔9 gibi simetri tuzakları |
| Ses | arka plan gürültüsü, hız değiştirme | aşırı hız değişimi fonemi bozar |
| Metin | eş anlamlı değiştirme, çevir-geri çevir | anlam kayması |
| **Titreşim / sensör** | gürültü ekleme, genlik ölçekleme, zaman kaydırma, pencere jitter'ı | **frekans içeriğini bozan dönüşümlerden kaçın** |

![Titreşim verisinde augmentation](/yz50/augmentation-titresim.png)

Titreşim verisinde mantık aynı: aynı arıza, farklı kayıt koşulu. Sensör biraz daha gürültülü
olabilir, bağlantı biraz gevşek olduğu için genlik farklı çıkabilir, kayıt birkaç milisaniye
kaymış olabilir. Bunların hepsi gerçek dünyada oluyor, dolayısıyla modelin bunlara dayanıklı
olması gerekiyor.

**Ama buradaki kritik sınır:** mekanik arıza teşhisi **belirli frekanslardaki tepelere**
dayanıyor. Zamanı esnetmek (time warping) o frekansları kaydırır — yani etiketi bozabilir.
Genlik ölçekleme ve gürültü ekleme güvenli, zaman esnetme değil.

## 3. Sızıntı tuzağı — sırayı karıştırırsan

**Bu bölüm neden var:** Augmentation'ın en sık yapılan hatası teknik değil, **sıra** hatası.
Sonucu da sessiz: model bozulmuyor, **test skoru yalan söylüyor**.

Yanlış sıra: önce bütün veriyi çoğalt, **sonra** train/test diye böl. Bu durumda aynı örneğin
orijinali eğitime, dönüştürülmüş kopyası teste düşebiliyor — model testte aslında daha önce
gördüğü bir şeyi tanıyor.

Doğru sıra: **önce böl**, sonra yalnızca eğitim tarafını çoğalt.

### Bunun adı var: bağımlı örnekler

Bu, augmentation'a özgü bir tuhaflık değil; istatistiksel olarak tanımlı bir durum.
scikit-learn dokümantasyonu sorunu şöyle koyuyor:

> *"The i.i.d. assumption is broken if the underlying generative process yields groups of
> dependent samples."*

Verilen örnek bizim durumumuzun birebir karşılığı: bir hastadan alınan **birden çok ölçüm**.
Bu ölçümler bağımsız değil, aynı kişiye ait. Dokümantasyonun istediği şey:

> *"we need to ensure that all the samples in the validation fold come from groups that are not
> represented at all in the paired training fold."*

Ve neden önemli olduğu:

> *"if the model is flexible enough to learn from highly person specific features it could fail
> to generalize to new subjects."*

Augmentation'la üretilen kopyalar da tam olarak böyle bir **grup** oluşturuyor: orijinal ve
onun bütün varyasyonları aynı gruba ait. Bölme grup seviyesinde yapılmazsa, ölçtüğün şey
"yeni veriye genelleme" değil, "aynı örneğin başka bir kopyasını tanıma" oluyor.

Araç olarak scikit-learn'de `GroupKFold`, `GroupShuffleSplit`, `LeaveOneGroupOut` bunun için var:
aynı grubun hem eğitimde hem testte görünmesini engelliyorlar.

Aynı doküman ön işleme için de aynı kuralı koyuyor — normalizasyon, ölçekleme, feature seçimi
**eğitim tarafından öğrenilip** teste uygulanmalı; `Pipeline` bunu otomatikleştirmek için var.

### Senin için doğrudan bağlantısı

Titreşim verisinde kayan pencere (sliding window) kullanmak **tam olarak bu grup yapısını**
üretiyor: örtüşen komşu pencereler aynı ham örnekleri paylaşıyor, yani bağımsız değiller.
Dolayısıyla bölme pencere seviyesinde değil, **kayıt/çalışma seviyesinde** olmalı — bir rulmanın
kaydı ya tamamen eğitimde ya tamamen testte.

Bu, Tiremo (UBMK 2025) bildirisinde kayda geçirilen şüphenin aynısı: 429.420 örnek tek bir ocak
demosundan kayan pencereyle üretilmiş ve bildiri bölmenin gruplu olup olmadığını yazmıyor.

> **Pratik işaret:** Gruplu bölmeye geçtiğinde skorun belirgin biçimde **düşmesi** beklenen
> şeydir — önceki ölçümün sızdırdığını gösterir. Ve mükemmele yakın bir test skoru (%100 gibi)
> bir başarı değil, önce sızıntıdan şüphelenmeni gerektiren bir **alarmdır**.

## 4. Otomatik hiperparametre arama

Elle ayarlamak sezgi kazandırıyor ([26](/posts/yz50-26-hiperparametre-ve-early-stopping/)) ama yavaş.
Üç otomatik yaklaşım var.

### Grid search — ızgara

Her hiperparametre için birkaç değer seç, **bütün kombinasyonları** dene.
Örnek: $\eta \in \{0.01,\ 0.1,\ 0.5\}$ ve $\lambda \in \{0.1,\ 1.0,\ 5.0\}$ → $3 \times 3 = 9$ deneme.

### Random search — rastgele örnekleme

Aralıkları belirle, o aralıklardan **rastgele** kombinasyonlar çek.

Bergstra & Bengio'nun 2012'deki gözlemi şu: **hiperparametreler eşit derecede önemli değil.**
`lr` sonucu belirler, bazı başka parametre neredeyse hiç etkilemez. Grid search bunu bilmediği
için, önemsiz eksende değer değiştirirken **önemli eksende aynı değerleri tekrar tekrar**
dener.

![Aynı bütçe, iki strateji](/yz50/grid-vs-random.png)

Yukarıdaki grafik **yapay bir örnek** — gerçek bir eğitim koşusu değil, Bergstra & Bengio'nun
anlattığı durumu görünür kılmak için kurulmuş bir çizim. Anlattığı şey şu:

- **Grid:** 3×3 ızgarada önemli eksende yalnızca **3 farklı** değer denenmiş oluyor; kalan
  6 deneme aynı üç değeri tekrarlıyor. Tepe o üç noktanın arasına düşerse bulunamıyor.
- **Random:** 9 denemenin hepsi önemli eksende **farklı** bir nokta, dolayısıyla tepeye
  yakın bir yere düşme şansı belirgin biçimde yüksek.

Asıl iddia ölçümden değil kaynaktan geliyor: Bergstra & Bengio (2012) bunu gerçek veri
setleri üzerinde göstermiş ve random search'ü aynı bütçede daha etkili bulmuştur.

### Bayesian optimization

Önceki denemelerin sonuçlarını kullanarak hiperparametre uzayının bir **modelini** kuruyor
(genelde Gaussian Process), sonra bir sonraki denemeyi bu modele göre seçiyor. İki şeyi
dengeliyor:

- **Exploitation:** şu ana kadar iyi çıkan bölgeyi daha ince taramak
- **Exploration:** hiç denenmemiş, belirsizliğin yüksek olduğu bölgeyi yoklamak

### Üçünün karşılaştırması

| | Güçlü | Zayıf | Ne zaman |
|---|---|---|---|
| **Grid** | Basit, tekrarlanabilir, paralelleştirmesi kolay, hangi değerin denendiği belli | Boyut arttıkça kombinasyon üstel patlıyor; önemsiz eksende kaynak israfı | 1–2 hiperparametre, ucuz model |
| **Random** | Aynı bütçeyle önemli ekseni çok daha zengin tarıyor; istediğin an durdurabilirsin | Tesadüfe bağlı; aynı bütçeyle iki koşu farklı sonuç verebilir | 3+ hiperparametre, varsayılan seçim |
| **Bayesian** | En az denemeyle en iyi noktaya gidiyor | Kendi hiperparametreleri var; kurulum maliyeti; paralelleştirmesi zor; ucuz modellerde getirisi masrafını karşılamıyor | Tek denemesi saatler/günler süren modeller |

Empa'nın UBMK bildirisinde **Optuna** geçiyor — Optuna, Bayesian optimization ailesinden
(TPE) bir kütüphane.

## 5. Pratik iş akışı

1. **Veri tarafı önce.** Mümkünse veri topla, mümkün değilse augmentation yap. Bu genelde
   hiperparametre kurcalamaktan daha çok kazandırıyor.
2. **Elle hızlı tur.** Problemi küçült, mertebeyi bul ([26](/posts/yz50-26-hiperparametre-ve-early-stopping/)
   bölüm 1–2). Bu adım sezgi kazandırıyor ve arama aralıklarını belirliyor.
3. **Otomatik ince ayar.** Belirlediğin aralıklarda random search; model çok pahalıysa Bayesian.
4. **Her aşamada bölme disiplinini koru.** Augmentation bölmeden sonra, karar validation'da,
   test en sonda bir kez.

## 6. Özet

- Augmentation = alan bilgini ("bu dönüşüm önemsiz") veriye gömmek.
- Geçerli dönüşüm **veriye bağlı** ve bunu model değil **sen** biliyorsun; 6↔9 ve titreşimde
  zaman esnetme uyarıcı örnekler.
- **Önce böl, sonra çoğalt.** Kopyalar bağımlı örnek grubu oluşturuyor; bölme grup
  seviyesinde yapılmazsa test skoru genellemeyi değil kopya tanımayı ölçüyor
  (scikit-learn: `GroupKFold`). Kayan pencerede grup = **kayıt**, pencere değil.
- Grid küçük uzayda, random varsayılan, Bayesian pahalı modellerde.
- Mükemmel test skoru bir başarı değil, bir **alarmdır**.

## Kaynaklar

- Michael Nielsen — [*Neural Networks and Deep Learning*, Bölüm 3](http://neuralnetworksanddeeplearning.com/chap3.html)
  (artificial expansion of training data)
- Bergstra & Bengio (2012) — [*Random Search for Hyper-Parameter Optimization*](https://www.jmlr.org/papers/v13/bergstra12a.html), JMLR
- Snoek, Larochelle & Adams (2012) — [*Practical Bayesian Optimization of Machine Learning
  Algorithms*](https://arxiv.org/abs/1206.2944), NIPS
- Simard, Steinkraus & Platt (2003) — esnek bozulmalar (elastic distortions), Nielsen'in atfettiği kaynak
- [scikit-learn — Cross-validation: grouped data](https://scikit-learn.org/stable/modules/cross_validation.html)
  (`GroupKFold`, bağımlı örnekler, `Pipeline` ile ön işleme sızıntısı)
- Bu nottaki grafikler: `img/make_augmentation_figures.py`

---

**Bağlantılı:** [26 · Hiperparametre Ayarlama ve Early Stopping](/posts/yz50-26-hiperparametre-ve-early-stopping/) · [25 · L1 ve L2 Regularization](/posts/yz50-25-regularization-l1-l2/) · [14 · Kaiming Init ve BatchNorm](/posts/yz50-14-kaiming-init-ve-batchnorm/)
