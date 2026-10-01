---
title: "25 · L1 ve L2 Regularization"
published: 2026-10-01
description: "L2 ağırlığı sıfıra yaklaştırır, L1 vardırır. Farkın tek sebebi itme kuvveti: biri ağırlıkla orantılı, diğeri sabit."
tags:
  - YZ50
  - Regularization
  - Optimizasyon
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

**Bu not neden var:** "L1 ile L2 farkı ne" sorusu iki ayrı yeri kastedebiliyor ve ikisi
karıştırılıyor: **loss fonksiyonu** olarak L1/L2 (bunu
[17 · MAE'nin Köşesi ve Huber Loss](/posts/yz50-17-mae-huber-loss/) zaten anlatıyor) ve
**regularization cezası** olarak L1/L2 (bu, notların hiçbirinde yoktu). Bu not ikincisini
sıfırdan kuruyor ve sonunda ikisini birbirine bağlıyor.

Konu ayrıca [08 · Stochastic Gradient Descent](/posts/yz50-08-stochastic-gradient-descent/) ve
[09 · Optimizer Varyantları](/posts/yz50-09-optimizer-varyantlari/) ile doğrudan ilişkili — `weight_decay`
denen şeyin ne olduğu ve Adam'da neden beklendiği gibi davranmadığı burada çözülüyor.

---

## 1. Önce problem: ağırlıkları neden küçük tutmak isteyelim

**Bu bölüm neden var:** Regularization'ı "genellemeyi iyileştiren bir hile" diye ezberlemek
kolay, ama o zaman L1 ile L2 arasında seçim yapamazsın. Önce cezanın hangi sorunu çözdüğünü
kurmak gerekiyor.

Bir modelin ağırlıkları çok büyüyebiliyorsa, model girdideki küçük oynamalara çok büyük
tepkiler verebilir demektir. Bu, eğitim verisindeki **gürültüyü de ezberleyebilmesi** anlamına
geliyor: gürültüye tam oturmak için ağırlıkların birbirini götüren büyük değerler alması gerekir.

Çözüm fikri basit: modele "hatayı küçült" demekle yetinme, **"ama ağırlıkları da küçük tut"**
diye ikinci bir istek ekle. Yani optimize ettiğin şeyi değiştir:

$$
\text{yeni loss} = \underbrace{L(w)}_{\text{veriye uy}} + \underbrace{\lambda \cdot R(w)}_{\text{ağırlıklar küçük kalsın}}
$$

Buradaki $\lambda$ iki isteğin ağırlığını ayarlayan sayı: büyükse "küçük kal" isteği baskın,
sıfırsa ceza hiç yok.

Geriye tek soru kalıyor: **"ağırlıklar küçük kalsın"ı matematiksel olarak nasıl yazacağız?**
İki doğal cevap var ve bütün mesele bu ikisinin farkında.

## 2. İki ceza

| Ad | Ceza terimi $R(w)$ | Diğer adı |
|---|---|---|
| **L2** | $\sum_i w_i^2$ — karelerin toplamı | Ridge, weight decay |
| **L1** | $\sum_i \lvert w_i \rvert$ — mutlak değerlerin toplamı | Lasso |

İsimler buradan geliyor: $\lVert w \rVert_2$ vektörün 2-normu (karelerin toplamının karekökü),
$\lVert w \rVert_1$ ise 1-normu (mutlak değerlerin toplamı). Ceza olarak kullanılırken L2'nin
karekökü alınmaz, doğrudan karelerin toplamı yazılır — türevi daha temiz olsun diye.

İlk bakışta ikisi aynı işi yapıyor gibi: ikisi de $w$ sıfırdan uzaklaştıkça büyüyor, yani ikisi
de ağırlığı küçük kalmaya teşvik ediyor. **Ama davranışları tamamen farklı**, ve fark cezanın
kendisinde değil, **cezanın türevinde**.

## 3. Asıl fark: itme kuvveti

**Bu bölüm neden var:** Bu notun tek cümlelik özeti burada. Geri kalan her şey bunun sonucu.

Gradient descent ağırlığı cezanın **türevi** kadar itiyor. İki cezanın türevini alalım:

$$
\frac{\partial}{\partial w}\left(\lambda w^2\right) = 2\lambda w
\qquad\qquad
\frac{\partial}{\partial w}\left(\lambda \lvert w \rvert\right) = \lambda \cdot \mathrm{sign}(w)
$$

Dikkat: **L2'nin itmesinde $w$ var, L1'inkinde yok.**

- **L2:** ağırlık büyükse sert, küçükse yumuşak iter. İtme kuvveti ağırlıkla orantılı.
- **L1:** ağırlık ne olursa olsun **aynı** kuvvetle iter. Sabit.

$\lambda = 0.01$ için sayılara bakalım:

| $w$ | L2 itmesi $2\lambda w$ | L1 itmesi $\lambda$ | L1 kaç kat daha sert |
|---|---|---|---|
| 1.0 | 0.02 | 0.01 | 0.5× (L2 daha sert) |
| 0.5 | 0.01 | 0.01 | **1× (eşit)** |
| 0.1 | 0.002 | 0.01 | 5× |
| 0.01 | 0.0002 | 0.01 | 50× |
| 0.001 | 0.00002 | 0.01 | 500× |

![L1'in itme kuvveti sabit, L2'ninki ağırlıkla küçülüyor](/yz50/ceza-itme-kuvveti.png)

Tablonun anlattığı şey şu: **ağırlık küçüldükçe L2 pes ediyor, L1 etmiyor.** $w = 0.001$'de
L2'nin itmesi neredeyse yok olmuş, L1 ise hâlâ ilk günkü kuvvetiyle itiyor.

Sonuç doğrudan buradan çıkıyor:

- **L2 ağırlığı sıfıra yaklaştırır ama hiçbir zaman sıfıra vardıramaz.** Yaklaştıkça itme
  zayıfladığı için, her adımda kalan mesafenin bir kısmını kapatır — sonsuza kadar yaklaşır,
  varmaz.
- **L1 ağırlığı sıfıra taşır ve orada bırakır.** İtme sabit olduğu için, ağırlık bir adımlık
  mesafeden daha yakına geldiğinde adım onu sıfırın öbür tarafına atar; oradan da geri iter.
  Ağırlık sıfırda sıkışıp kalır.

![Aynı ağırlık L1 ile sıfıra varıyor, L2 ile varamıyor](/yz50/agirlik-yolculugu.gif)

### Önemli ayrıntı: "tam sıfır" bedava gelmiyor

L1'in türevi $w = 0$ noktasında tanımsız (mutlak değerin köşesi —
[17 · MAE'nin Köşesi ve Huber Loss](/posts/yz50-17-mae-huber-loss/) notunda aynı köşe var). Ders kitaplarında
bu, $\mathrm{sgn}(0) = 0$ konvansiyonuyla geçiştiriliyor ve güncelleme şöyle yazılıyor:

$$
w \leftarrow w - \eta\,\lambda\,\mathrm{sgn}(w) - \eta\,\frac{\partial C_0}{\partial w}
$$

**Ama bu kural ağırlıkları tam sıfıra oturtmuyor.** Sabit itme, ağırlık her sıfırı geçtiğinde
işaret değiştirdiği için ağırlık sıfırın etrafında salınıyor — çok küçülüyor ama `0.0` olmuyor.
Aşağıdaki (bölüm 5) veri setinde ikisini de koşturdum:

| Yöntem | Tam `0.0` olan | $\lvert w \rvert < 10^{-3}$ | En küçük sıfır-dışı |
|---|---|---|---|
| Altgradyan ($\mathrm{sgn}(0)=0$, düz SGD) | **0 / 20** | 7 / 20 | $3.7 \times 10^{-5}$ |
| **Proximal** (soft-thresholding) | **17 / 20** | 17 / 20 | $0.861$ |

Tam sıfır, **proximal adım** uygulandığında çıkıyor: normal gradient adımı atılıp ardından

$$
w \leftarrow \mathrm{sign}(w)\,\max\!\left(\lvert w \rvert - \text{lr}\cdot\lambda,\ 0\right)
$$

Yani ağırlık, bir adımlık itmeden küçükse **sıfıra kıstırılıyor**. Bu nottaki animasyon ve
ölçüm bu adımı kullanıyor.

**Pratik sonucu:** pruning yaparken `w == 0` diye budarsan düz SGD'de **hiçbir şey budayamazsın**.
Ya proximal kullanacaksın ya da bir eşik koyacaksın ($\lvert w \rvert < 10^{-4}$ gibi).
scikit-learn'ün `Lasso`'su bu yüzden düz gradient descent değil **koordinat inişi** kullanıyor.

## 4. İkinci bakış: neden köşe sıfır üretiyor

**Bu bölüm neden var:** Aynı sonuca bir de geometriden varılıyor ve literatürde en çok bu resim
gösteriliyor. 3. bölümü anladıysan buna ihtiyacın yok — ama bu resim başka bir yerde karşına
çıktığında tanıyasın diye kuruyorum.

"Cezayı loss'a ekle" formülasyonu, şuna denk: **loss'u, ağırlık vektörü belli bir bölgenin
içinde kalmak şartıyla minimize et.**

$$
\min_w L(w) \quad \text{öyle ki} \quad \lVert w \rVert \le t
$$

İki ağırlıklı ($w_1, w_2$) bir modelde bu bölgeler şöyle:

- $\lvert w_1 \rvert + \lvert w_2 \rvert \le t$ → **elmas** (eksenler üzerinde dört köşesi var)
- $w_1^2 + w_2^2 \le t^2$ → **çember** (hiç köşesi yok)

Loss'un seviye eğrileri, cezasız çözümün etrafında büyüyen elipsler. Kısıtlı çözüm, bu
elipslerin bölgeye **ilk değdiği** nokta.

![Kısıt bölgesinin köşesi neden tam sıfır üretiyor](/yz50/elmas-vs-cember.png)

Elipsin elmasa ilk değdiği yer büyük olasılıkla bir **köşe** — çünkü köşe dışarı doğru çıkıntı
yapıyor, elips oraya diğer kenarlardan önce ulaşıyor. Ve **köşeler eksenlerin üzerinde**, yani
o noktada diğer koordinat tam olarak sıfır. Yukarıdaki soldaki grafikte çözüm $(0.75,\ 0.00)$
çıktı: $w_2$ tam sıfır.

Çemberde hiçbir nokta ayrıcalıklı değil, elips çembere herhangi bir yerinden değebiliyor ve o
yerin eksen üzerine düşmesi için özel bir sebep yok. Sağdaki grafikte çözüm $(0.60,\ 0.44)$ —
ikisi de sıfırdan farklı.

Yani sparsity bir sezgi değil, **kısıt bölgesinin köşeli olmasının doğrudan sonucu**.

## 5. Ölçüm: 20 ağırlıklı bir deney

**Bu bölüm neden var:** Yukarıdaki iki argüman da teorik. Gerçekten öyle olup olmadığını
ölçmeden yazmak, bu not setinde daha önce yedi hataya yol açmış bir alışkanlık.

Kurulum: 20 özellik, 220 örnek. Ama veriyi üretirken **sadece 3 özelliğe** gerçek katsayı
verildi (indeks 2, 7, 13), kalan 17'sinin gerçek katsayısı sıfır. Yani doğru cevap belli:
iyi bir yöntem 17 ağırlığı sıfırlamalı.

![20 ağırlıkta L1 ve L2'nin bıraktığı sonuç](/yz50/sparsity-sonuc.png)

Ölçülen sonuç:

| | Tam sıfır olan ağırlık | Hangilerini tuttu |
|---|---|---|
| **L2** | **0 / 20** | hepsini tuttu, sadece küçülttü |
| **L1** | **17 / 20** | tam olarak 2, 7 ve 13 — doğru olanlar |

L1 hiçbir yardım almadan, gerçekten işe yarayan üç özelliği buldu ve diğerlerini tam sıfıra
indirdi. L2 ise işe yaramayan ağırlıkları küçülttü ama hiçbirini sıfırlamadı — grafikte
küçük ama sıfır olmayan çubuklar olarak duruyorlar.

> **Kapsam uyarısı:** Bu tek bir sentetik veri seti ve tek bir $\lambda$ değeri (0.35). Sayılar
> "L1 her zaman doğru özellikleri bulur" demek değil — özellikler birbiriyle korelasyonluysa
> L1'in davranışı bozuluyor (bkz. bölüm 7, Elastic Net). Gösterdiği şey, sıfırlama davranışının
> gerçekten olduğu. Üreten kod: `img/make_regularization_figures.py`.

## 6. Nerede hangisi kullanılır

| İhtiyaç | Seçim | Neden |
|---|---|---|
| Genel amaçlı, derin öğrenmede varsayılan | **L2** | Ağırlıkları küçük tutuyor, modele yapı dayatmıyor, türevi her yerde tanımlı |
| Hangi özellikler gerçekten işe yarıyor bilmek | **L1** | Gereksizleri tam sıfıra indirip kendiliğinden feature selection yapıyor |
| Modeli küçültmek (**pruning**) | **L1** | Sıfır olan ağırlıklar tamamen atılabiliyor → daha küçük model |
| Korelasyonlu özellikler var | **Elastic Net** | Bölüm 7 |

**Pruning bağlantısı önemli:** TinyML'de modeli mikrodenetleyiciye sığdırmanın iki ayrı kolu
var — **quantization** (her sayıyı daha az bitle tut) ve **pruning** (bazı ağırlıkları tamamen
at). L1 ikincisini mümkün kılıyor: sıfırlanan ağırlıklar taşınmıyor, saklanmıyor, çarpılmıyor.
İkisi birbirini dışlamıyor, üst üste uygulanabiliyor.

## 7. Elastic Net — L1'in zayıf noktası

İki özellik birbiriyle güçlü korelasyonluysa (neredeyse aynı bilgiyi taşıyorsa), L1 genelde
**birini keyfi olarak seçip diğerini sıfırlıyor** — hangisini seçtiği veri biraz değişince
değişebiliyor, bu da yorumu kırılgan yapıyor. L2 ise böyle bir durumda ağırlığı ikisine
paylaştırıyor.

Elastic Net ikisini birden kullanıyor:

$$
\lambda_1 \sum_i \lvert w_i \rvert + \lambda_2 \sum_i w_i^2
$$

L1 terimi sparsity'yi getiriyor, L2 terimi korelasyonlu özellikleri grup halinde tutup seçimi
kararlı kılıyor.

## 8. `weight_decay` ve Adam tuzağı

**Bu bölüm neden var:** PyTorch'ta L2 regularization'ı optimizer'ın `weight_decay` parametresiyle
veriyorsun. Ama bu parametre Adam'da beklediğin şeyi yapmıyor ve bu, pratikte sık karşılaşılan
gerçek bir hata kaynağı. [09 · Optimizer Varyantları](/posts/yz50-09-optimizer-varyantlari/) Adam'ı
anlatıyor ama bu ayrıntıya girmiyordu.

Sade gradient descent'te "L2 cezası eklemek" ile "her adımda ağırlığı biraz küçültmek" aynı
şey — adı buradan geliyor, **weight decay**.

Ama Adam gradient'i $\sqrt{\hat v}$'ye bölerek **her ağırlık için ayrı ayrı ölçekliyor**
(09'daki adaptif adım fikri). `weight_decay` parametresi cezayı gradient'in içine eklediği
için, ceza da bu ölçeklemeden geçiyor. Sonuç: **büyük gradient alan ağırlıklar daha az
cezalanıyor** — oysa regularization'dan beklediğin şey tam tersi, herkese eşit davranması.

Çözüm **AdamW**: weight decay'i gradient'ten ayırıp (decoupled) doğrudan ağırlığa uyguluyor.
Loshchilov & Hutter bunu 2017'de ortaya koydu ve modern mimarilerde varsayılan hale geldi.

```python
# L2 cezası gradient'in içine giriyor — Adam'ın ölçeklemesinden etkileniyor
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

# weight decay ayrı uygulanıyor — istediğin davranış bu
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

**Pratik kural:** Adam kullanıyorsan ve regularization istiyorsan `AdamW` kullan.

> **Kapsam uyarısı:** Bu, "AdamW her zaman daha iyi sonuç verir" demek değil — iddia, weight
> decay'in Adam'da niyet edilen etkiyi göstermediği ve AdamW'nin bunu düzelttiği. Hangisinin
> daha iyi genellediği probleme ve $\lambda$ ayarına bağlı.

## 9. Loss olarak L1/L2 ile bağlantı

Aynı iki isim loss fonksiyonu olarak da geçiyor ve **tamamen farklı bir şeyi** ölçüyor:
regularization ağırlıklara bakıyor, loss ise tahmin hatasına.

| | L2 | L1 |
|---|---|---|
| **Loss olarak** | MSE: $\sum (y - \hat y)^2$ | MAE: $\sum \lvert y - \hat y \rvert$ |
| Hangi istatistiği bulur | **ortalama** | **medyan** |
| Outlier'a karşı | hassas | dayanıklı |
| **Regularization olarak** | $\sum w_i^2$ — küçültür | $\sum \lvert w_i \rvert$ — sıfırlar |

Ama ortak bir sebep var, ve bu notun geri kalanı zaten onu kurdu: **karesel terimin türevi
değişkenle orantılı, mutlak değerinki sabit.** Loss tarafında bu, MSE'nin büyük hatalara
(outlier'lara) orantısız tepki vermesi demek; regularization tarafında ise L2'nin küçük
ağırlıkları rahat bırakması demek. Aynı matematik, iki farklı yerde.

Loss tarafının detayı — sıfırdaki köşe, Huber'in bunu nasıl yumuşattığı, hangi loss'un
outlier'a ne kadar dayandığının ölçümü — [17 · MAE'nin Köşesi ve Huber Loss](/posts/yz50-17-mae-huber-loss/) notunda.

## 10. Özet

- Regularization = loss'a "ağırlıklar küçük kalsın" isteğini eklemek.
- L2 karelerin toplamını, L1 mutlak değerlerin toplamını cezalandırıyor.
- **Tek kritik fark:** L2'nin itme kuvveti ağırlıkla orantılı ($2\lambda w$), L1'inki sabit
  ($\lambda$). Ağırlık küçülünce L2 pes ediyor, L1 etmiyor.
- Bu yüzden L2 sıfıra yaklaştırır, **L1 sıfıra vardırır** — ölçümde 17/20 ağırlık tam sıfır.
- Geometrik karşılığı: L1'in kısıt bölgesi köşeli ve köşeler eksenler üzerinde.
- Varsayılan L2; sparsity/feature selection/pruning gerekiyorsa L1; korelasyon varsa Elastic Net.
- Adam ile regularization istiyorsan **AdamW** kullan.

## Kaynaklar

- Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning*, Bölüm 3.4
  (Ridge ve Lasso'nun standart referansı, elmas/çember resmi de oradan)
- Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso*
- Loshchilov & Hutter (2017) — [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (AdamW)
- [PyTorch — `torch.optim` dokümantasyonu](https://docs.pytorch.org/docs/stable/optim.html)
  (`Adam` ve `AdamW`'nin `weight_decay` davranışı)
- Bu nottaki sayılar ve grafikler: `img/make_regularization_figures.py`

---

**Bağlantılı:** [17 · MAE'nin Köşesi ve Huber Loss](/posts/yz50-17-mae-huber-loss/) · [09 · Optimizer Varyantları](/posts/yz50-09-optimizer-varyantlari/) · [08 · Stochastic Gradient Descent](/posts/yz50-08-stochastic-gradient-descent/)
