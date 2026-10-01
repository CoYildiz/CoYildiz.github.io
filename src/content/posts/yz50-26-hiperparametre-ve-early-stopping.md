---
title: "26 · Hiperparametre Ayarlama ve Early Stopping"
published: 2026-10-01
description: "'Öğrenme sinyali', 'kayıp düşüyorsa' ve 'overfit oldu' ifadelerinin her birini somut bir ölçüte bağlıyoruz."
tags:
  - YZ50
  - Hiperparametre
  - Overfitting
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

**Bu not neden var:** Hiperparametre anlatımlarında üç ifade sürekli geçiyor ve üçü de tanımsız
bırakılıyor: *"öğrenme sinyalini yakala"*, *"kayıp düşüyorsa"*, *"model overfit oldu"*. Bu not
üçünü de somutlaştırıyor: ne demek, neye bakılır, nasıl karar verilir.

Kaynak: Nielsen Bölüm 3 + 2026-10-01'de sorulan üç soru.

---

## 1. "Öğrenme sinyali" ne demek

Eğitime başlamadan önce ağın ağırlıkları **rastgele**. Rastgele bir ağ ne yapar? Rastgele tahmin.
10 sınıflı bir problemde her sınıfa eşit olasılık verir, ve bunun loss karşılığı sabit bir sayıdır:

$$
\text{rastgele tahminin loss'u} = \ln 10 = 2.30
$$

Eğitim başladığında yalnızca iki şey olabilir:

- **Loss 2.30 civarında kalır** → ağ rastgele tahminden daha iyisini yapmıyor. **Sinyal yok.**
- **Loss 2.30'un altına iner** → ağ veriden bir şey yakaladı. **Sinyal var.**

![Öğrenme sinyali: loss rastgele çizgisinden kopuyor mu](/yz50/ogrenme-sinyali-nedir.png)

İşte "öğrenme sinyalini yakalamak" bu ikinci durumu **ilk kez görmek** demek. Modelin iyi olması
gerekmiyor, doğruluğun yüksek olması gerekmiyor — sadece bir şeyin olduğunu görmek.

> **Radyo benzetmesi:** Radyoyu ayarlarken önce cızırtı duyarsın. "Sinyal yakalamak", cızırtının
> arasından müziğin ilk kez duyulduğu an — henüz net değil ama orada bir şey var. İnce ayarı
> ondan sonra yaparsın. Burada da aynısı: önce bir şeyin öğrenildiğini gör, sonra iyileştir.

### Rastgele çizgisi her problemde farklı

| Problem | Rastgele doğruluk | Rastgele loss |
|---|---|---|
| 2 sınıf (dengeli) | %50 | $\ln 2 = 0.69$ |
| 10 sınıf (dengeli) | %10 | $\ln 10 = 2.30$ |
| 10 sınıf, biri verinin %90'ı | **%90** | — |

Son satır önemli: sınıflar dengesizse çizgi %10 değil, **en büyük sınıfın oranı** olur.
[03 · Sınıflandırmaya Geçiş](/posts/yz50-03-siniflandirmaya-gecis/)'teki MNIST deneyinde tek bir rakamı
"hayır" diye tahmin eden model %90 doğruluk alıyordu — hiçbir şey öğrenmeden.

### Neden önce problemi küçültüyoruz

Karmaşık bir ağda sinyal göremediğinde iki ihtimal var ve **ayırt edemezsin**:

1. Hiperparametreler kötü
2. Kodda hata var

Bu yüzden Nielsen önce problemi soyup küçültmeyi söylüyor: 10 sınıf yerine 2 sınıf, gizli katman
yok, validation seti 100 örnek. Amaç iyi model eğitmek değil — **çalışmak zorunda olan** bir
kurulum elde etmek. Sinyali orada gördüğünde temelin sağlam olduğunu bilirsin; karmaşıklığı
sonra tek tek geri eklersin ve hangisi bozarsa onu bilirsin.

### Sinyali aramanın en ucuz yolu: 10 örneği ezberlet

YZ50'de de karşına çıkan adım — Hafta 4 Görev 3'teki *"önce tek minibatch'i overfit et"*.
Mantık şu: doğru kurulmuş bir ağ, avuç içi kadar veriyi **ezberleyebilmek zorundadır**.
10 örnek için 48 nöronluk bir ağın kapasitesi fazlasıyla yeter. Loss sıfıra inmiyorsa sorun
hiperparametrede değil, kurulumda.

![Tek batch testi neyi yakalar, neyi yakalamaz](/yz50/ogrenme-sinyali-tek-batch.png)

| Kurulum | 200 epoch sonunda loss | |
|---|---|---|
| Doğru kurulmuş ağ | $3.1 \times 10^{-3}$ | ezberledi |
| Etiketler rastgele karıştırılmış | $4.1 \times 10^{-3}$ | **yine ezberledi** |
| Gradient işareti ters | $27.6$ | patladı, hata yakalandı |
| `lr` $= 10^{-6}$ | $2.29$ | $\ln 10$'da takılı, hiç kıpırdamadı |

İkinci satır şaşırtıcı görünebilir: etiketleri **rastgele karıştırdığım** ağ da ezberledi.
Bu bir aksaklık değil — yeterli kapasiteli bir ağ anlamsız etiketleri de ezberler (Zhang ve ark.
2017 bunu tam MNIST'te gösterdi).

**Testin güçlü ve zayıf yanları:**

| | |
|---|---|
| **Yakalar** | Gradient formülü/işareti hatası, kopuk backward, ölü learning rate, yanlış tensor shape, doygun aktivasyon |
| **Yakalamaz** | Etiketlerin yanlış hizalanması, veri sızıntısı, hatalı ön işleme |
| **Maliyeti** | Saniyeler — bütün veriyle saatlerce eğitip "neden öğrenmiyor" demekten çok ucuz |
| **Sınırı** | Geçmesi modelin **iyi** olduğunu göstermez, sadece **bozuk olmadığını** gösterir |

Yakalamadığı şeyi yakalamanın yolu bu test değil: birkaç örneği etiketiyle birlikte **gözle**
kontrol etmek.

## 2. "Kayıp düşüyorsa" — hangi kayıp

Learning rate ararken "kayıp düşüyorsa artır" deniyor ama hangi kayıp olduğu söylenmiyor.
Cevap: **eğitim verisi üzerindeki loss**, ve **ilk birkaç epoch'ta**.

Neden validation accuracy değil:

- Bu aşamadaki soru *"model genelliyor mu"* değil, **"optimizasyon çalışıyor mu"**. Farklı sorular.
- Validation accuracy gürültülü ve geç tepki veriyor; loss sürekli ve hemen tepki veriyor.
- Accuracy basamaklı bir büyüklük — [03 · Sınıflandırmaya Geçiş](/posts/yz50-03-siniflandirmaya-gecis/) notunda
  türevinin neden kullanılamadığını türetmiştik. Aynı sebeple ince ayarda kötü gösterge.

![Çok küçük / iyi / sınırda / iraksayan learning rate](/yz50/lr-esigi-tarama.png)

| `lr` | 40 epoch sonunda eğitim loss'u | Görünüm |
|---|---|---|
| 0.001 | 2.12 | neredeyse düz — çok küçük |
| **0.05** | **0.10** | düzenli iniyor — **iyi** |
| 1.5 | 2.30 | salınıyor, ilerlemiyor |
| 6.0 | 2.40 | patlıyor |

Aradığın şey: loss'un ilk epoch'lardan itibaren **düzenli indiği en büyük mertebe**. Bulunca
yarısını al — tam sınırda çalışmak eğitimin ilerleyen aşamalarında kırılgan oluyor.

**Bu eşiğin matematiği sende zaten var.** [01 · Lineer Regresyon](/posts/yz50-01-lineer-regresyon/) notunun
4. bölümünde $\mathrm{lr} < 2/\lambda_{\max}$ koşulunu türetmiştik; orada ölçülen ilk ıraksama
$0.004593$, teorinin verdiği $0.004555$'ten %0.83 farklıydı. Yukarıdaki "salınım başlıyor"
noktası aynı olgu. Tek fark: derin ağda $\lambda_{\max}$'ı hesaplayamadığın için **tarayarak**
buluyorsun.

**Taramanın güçlü ve zayıf yanları:**

| | |
|---|---|
| **Güçlü** | Hiçbir teorik hesap gerektirmiyor, her mimaride çalışıyor, birkaç dakika sürüyor |
| **Zayıf** | Bulduğun eşik **o veriye ve o mimariye özgü** — veri ölçeğini değiştirirsen eşik kayar |
| **Zayıf** | Eğitimin başındaki eğriye bakıyor; başta iyi olan bir `lr` sonradan fazla büyük kalabilir |

Son maddenin çözümü **learning rate schedule**: başta büyük adım, validation plato yapınca
`lr`'yi yarıya (ya da 1/10'a) düşür.

## 3. Model gerçekten overfit oldu mu

En yaygın tanım "eğitim doğruluğu yüksek, test düşük" — ve bu tanım **yanıltıcı**. Aradaki fark
tek başına overfitting demek değil.

**Doğru ölçüt:** eğitim loss'u düşmeye devam ederken **validation başarımının iyileşmeyi
bırakması**.

![Overfitting'in teşhisi gap değil, validation'ın durması](/yz50/overfit-teshisi.png)

150 örnekle kasıtlı overfit ettirdim, 300 epoch:

| | |
|---|---|
| Validation zirvesi | **epoch 29**, %91.4 |
| Son epoch'ta validation | %90.7 |
| Zirvede eğitim doğruluğu | %98.7 |
| Eğitim loss'u, zirve → son | $0.042 \rightarrow 0.0013$ (**32 kat** iyileşme) |

Epoch 29'dan sonra eğitim loss'u 32 kat daha iyileşti, validation hiç iyileşmedi. O noktadan
sonraki bütün iş eğitim verisine özgü ayrıntıları ezberlemeye gitti.

**Üç ayrım:**

1. **Gap ≠ overfitting.** Zirvede bile %98.7 / %91.4 farkı var. Fark her zaman olur. Önemli olan
   validation'ın hâlâ yükselip yükselmediği. Yükseliyorsa **devam et**.
2. **İkisi de kötüyse** overfitting değil — underfitting ya da bozuk kurulum. Bölüm 1'e dön.
3. **Aynı metriği karşılaştır.** Eğitimde loss, validation'da accuracy'ye bakıp "ters gidiyorlar"
   demek yanıltıcı; ikisini de her iki küme için takip et.

### Neden test değil validation

Durdurma kararını ya da hiperparametre seçimini **test seti üzerinde** verirsen, test setine
hiperparametre düzeyinde uyum sağlarsın — test artık tarafsız bir tahmin vermez.

Bölüm: parametreler **training** ile öğrenilir · hiperparametreler ve durdurma kararı
**validation** ile verilir · **test** yalnızca en sonda, bir kez.

## 4. Early stopping

Grafiğe sonradan bakıp "burada durmalıydım" demek kolay; eğitim sırasında zirvenin zirve olduğunu
bilemezsin. Bu yüzden kural gerekiyor.

**no-improvement-in-$n$:** validation başarımı son $n$ epoch boyunca yeni bir en iyi değere
ulaşmadıysa dur, ve **en iyi değerin elde edildiği andaki ağırlıkları** geri yükle.

Yukarıdaki ölçümde $n = 20$ ile epoch 49'da durulur, geri yüklenen ağırlıklar epoch 29'unki olurdu.

| | |
|---|---|
| **Güçlü** | Ücretsiz — ek hesap yok, ek hiperparametre neredeyse yok. Hem zaman hem overfitting kazancı |
| **Güçlü** | Kaç epoch eğiteceğini önceden bilme zorunluluğunu kaldırıyor |
| **Zayıf** | Ağlar bazen uzun plato yapıp sonra tekrar ilerliyor; küçük $n$ bunu kaçırır |
| **Zayıf** | Validation gürültülüyse "yeni en iyi" rastlantısal olabilir — küçük validation setinde riskli |

Pratik: keşif aşamasında agresif ($n = 10$), nihai modelde esnek ($n = 20$–$50$).

## 5. $\lambda$ seçimi — ve $n$ ile ölçeklemedeki yaygın hata

$\lambda$ aramanın pratiği:

1. Önce $\lambda = 0$ ile uygun $\eta$'yı bul (bölüm 2).
2. $\lambda = 1.0$'dan başlayıp **10'ar kat** büyüt/küçült, validation'a bak — mertebeyi bul.
3. Mertebe oturunca ince ayar yap ve $\eta$'yı yeniden dengele (ikisi etkileşiyor).

### Yaygın gerekçe hatası

Nielsen'in L2 anlatımında $n$ 1.000'den 50.000'e çıkınca $\lambda$ 0.1'den 5.0'e çıkarılıyor.
Gerekçe olarak genelde şu söyleniyor: *"weight decay çarpanı $1 - \eta\lambda/n$ olduğundan,
$\lambda$ sabit kalsa etki 50 kat zayıflardı."*

**Bu gerekçe adım başına doğru, epoch başına yanlış.** Bir epoch'ta $n/m$ güncelleme var;
$n$ 50 kat büyüyünce güncelleme sayısı da 50 kat artıyor. Hesap ($\eta = 0.5$, $m = 10$):

| Kurulum | Adım başına çarpan | Adım / epoch | **Epoch sonu çarpan** |
|---|---|---|---|
| $n=1.000,\ \lambda=0.1$ | 0.99995 | 100 | **0.99501** |
| $n=50.000,\ \lambda=0.1$ (sabit) | 0.999999 | 5.000 | **0.99501** — aynı |
| $n=50.000,\ \lambda=5.0$ | 0.99995 | 5.000 | **0.77880** |

$\lambda$ sabit bırakılsaydı epoch başına etki **birebir korunuyordu**. $\lambda = 5.0$ aynı
baskıyı korumuyor, epoch başına kabaca 50 kat güçlendiriyor.

Objektiften de aynı sonuç: $C = C_0 + \frac{\lambda}{2n}\sum w^2$'de hem veri terimi hem ceza
$1/n$ taşıyor, dolayısıyla cezanın **göreli** ağırlığı zaten $\lambda$ — $n$'den bağımsız.

$\lambda = 5.0$'ın daha iyi sonuç vermesi gerçek bir **deneysel bulgu**; "aynı baskıyı korumak
için matematiksel zorunluluk" değil.

> **Doğrulanamayan kısım:** Bu gerekçenin Nielsen'in kendi metnine mi yoksa ikincil özetlere mi
> ait olduğu doğrulanamadı — kitabın sitesinde sertifika hatası var, PDF'ten yalnızca içindekiler
> alınabildi. $\lambda = 5.0$ kullandığı kesin; gerekçeyi onun kurup kurmadığı açık soru.

## 6. Mini-batch boyutu

Diğer hiperparametrelerden görece bağımsız, o yüzden ayrı ayarlanabiliyor.

| $m$ | Güçlü | Zayıf |
|---|---|---|
| **1** (online) | Her örnekten hemen öğreniyor; veri akış halindeyse değerli | Donanımın matris paralelliğini kullanamıyor; gradient çok gürültülü |
| **Orta** (16–256) | Gürültü/hız dengesi; donanım verimli | — |
| **Çok büyük** | Gradient tahmini çok temiz; donanım tam dolu | Güncelleme sıklığı düşüyor; aynı ilerleme için daha çok veri geçiyor |

Seçim ölçütü önemli: **epoch sayısına değil, geçen gerçek süreye karşı** validation başarımını
çiz. Epoch başına karşılaştırma büyük batch'i haksız yere iyi gösterir — bir epoch'ta daha az
güncelleme yaptığını gizler.

Batch boyutunun gürültü/hız dengesi [08 · Stochastic Gradient Descent](/posts/yz50-08-stochastic-gradient-descent/) notunda.

## 7. Özet

| Soru | Neye bakılır | Karar |
|---|---|---|
| Öğrenme sinyali var mı | Eğitim loss'u, rastgele çizgisine göre | Çizgiden kopmuyorsa kurulum bozuk olabilir |
| Hangi kayıp düşüyor | **Eğitim** loss'u, ilk epoch'lar | Düzenli indiği en büyük mertebe → yarısı |
| Overfit mi | Eğitim loss'u **ve** validation başarımı | Eğitim düşerken validation durduysa evet |

- Tek-batch testi tesisat hatasını yakalar, **etiket hatasını yakalamaz**.
- Gap tek başına overfitting değil; validation yükseliyorsa devam.
- Durdurma kararı **validation** ile, test en sonda bir kez.
- $\lambda$'yı $n$ ile ölçeklemenin "aynı baskı" gerekçesi epoch başına geçersiz.

## Kaynaklar

- Michael Nielsen — [*Neural Networks and Deep Learning*, Bölüm 3](http://neuralnetworksanddeeplearning.com/chap3.html)
- Zhang, Bengio, Hardt, Recht & Vinyals (2017) — [*Understanding Deep Learning Requires
  Rethinking Generalization*](https://arxiv.org/abs/1611.03530)
- Bu nottaki sayılar ve grafikler: `img/make_tuning_figures.py`, `img/make_signal_figure.py`

---

**Bağlantılı:** [25 · L1 ve L2 Regularization](/posts/yz50-25-regularization-l1-l2/) · [27 · Data Augmentation ve Otomatik Hiperparametre Arama](/posts/yz50-27-augmentation-ve-otomatik-arama/) · [01 · Lineer Regresyon](/posts/yz50-01-lineer-regresyon/) · [03 · Sınıflandırmaya Geçiş](/posts/yz50-03-siniflandirmaya-gecis/) · [08 · Stochastic Gradient Descent](/posts/yz50-08-stochastic-gradient-descent/)
