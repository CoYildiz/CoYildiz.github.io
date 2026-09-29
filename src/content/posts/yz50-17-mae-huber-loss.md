---
title: "17 · MAE'nin Köşesi ve Huber Loss"
published: 2026-09-29
description: "MAE'nin sıfırdaki köşesi neden sorun, Huber loss bunu nasıl yumuşatıyor ve outlier'a karşı hangi loss gerçekte ne kadar dirençli."
tags:
  - YZ50
  - Loss
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

İki soruya cevap: mutlak değerin köşesinde derivative'i elle "seçmek" sorunu çözüyor mu, ve Huber
aykırı değerleri gerçekte nasıl ayırıyor.

Notasyon: $u = \hat{y} - y$, yani artık. Bütün loss'lar tek bir örnek için yazılıyor,
batch hali bunların ortalaması.

---

## 1. Köşede derivative'i 0 seçmek fonksiyonu pürüzsüz yapmıyor

**Bu bölüm neden var:** MAE'nin köşesinde derivative tanımsız, ve kütüphaneler oraya keyfi bir değer (genelde 0) koyuyor. "Sorun çözüldü" gibi duruyor — çözülmediğini göstermek gerekiyor, çünkü asıl etki noktanın kendisinde değil komşuluğunda.

$|u|$ fonksiyonu $u=0$'da derivativelenemez, çünkü iki taraflı limit uyuşmuyor:

$$\lim_{h \to 0^+} \frac{|h|}{h} = +1, \qquad \lim_{h \to 0^-} \frac{|h|}{h} = -1$$

Buraya bir değer atamak — PyTorch `sign(0) = 0` diyor — limiti var etmiyor. **Differentiablelik
bir değere sahip olmak değil, limitin var olması demek.** Atadığımız şey bir sözleşme:
konveks fonksiyonların köşesinde tek bir derivative yerine bir **altderivative kümesi** var,

$$\partial |u|\big|_{u=0} = [-1, +1]$$

ve bu kümeden herhangi bir eleman seçmek subgradient descent'in yakınsaklık teoremlerini
bozmuyor. Seçim geçerli, ama fonksiyonu pürüzsüz yapmıyor.

### Ayrım: $C^0$ ve $C^1$

- $|u|$ **süreklidir** ($C^0$) — grafikte kopukluk yok.
- Derivative'i $\operatorname{sign}(u)$ ise $u=0$'da **süreksizdir**, yani $|u|$ sürekli derivativeli
  ($C^1$) değildir. Tek bir noktaya değer atamak süreksiz bir fonksiyonu sürekli yapmaz.

Sürekli olmak ile differentiable olmanın ayrıldığı en temiz örnek bu.

### Asıl bedel noktada değil, komşuluğunda

Tam $u=0$'a düşme ihtimali ölçüm-sıfır bir olay; float aritmetiğinde pratikte olmuyor.
Sorun köşenin **etrafında**: gradient'in büyüklüğü hatadan bağımsız olarak hep $1$.

| | gradient | hata küçülürken ne oluyor |
|---|---|---|
| MSE | $2u$ | gradient da küçülüyor, adımlar kendiliğinden kısalıyor |
| MAE | $\operatorname{sign}(u)$ | gradient sabit $\pm 1$, adım hep tam boy |
| Huber | $\operatorname{clip}(u, -\delta, \delta)$ | $\vert u\vert < \delta$ bölgesinde MSE gibi davranıyor |

Sabit learning rate ile MAE minimumun etrafında **duramaz** — her adım aynı boyda olduğu için
sürekli üstünden atlar. Yakınsaması için lr'yi zamanla düşürmen gerekir; subgradient descent'in
yakınsaklık garantisi zaten azalan adım boyu şartıyla geliyor.

![Aynı lr ile MAE salınıyor, Huber yakınsıyor](/yz50/huber-neden-kink-onemli.gif)

Aynı başlangıç ($u_0 = 2.5$), aynı learning rate (0.3). 22. adımda:

- **MAE:** son 5 adımın genliği **0.300** — yani hâlâ tam boy adımlarla $+0.1$ ile $-0.2$
  arasında gidip geliyor. Sonsuza kadar da böyle gider.
- **Huber:** son 5 adımın genliği **0.007**, $u = +0.002$. Durmuş.

Fark algoritmada değil, $|u| < \delta$ bölgesinde gradient'in hataya orantılı olmasında.

---

## 2. Huber aykırı değeri "ayırmıyor", etkisini tavanlıyor

**Bu bölüm neden var:** Huber sık sık "aykırı değerleri atan loss" diye anlatılıyor. Atmıyor — hiçbir noktayı dışlamıyor, sadece hiçbirinin katkısının bir tavanı geçmesine izin vermiyor. Aradaki fark önemli.

Burada kafa karışıklığının kaynağı şu beklenti: Huber'in bir yerde "bu nokta aykırı" diye karar
verdiği, onu dışladığı. Öyle bir karar yok. Olan şey daha basit: **her noktanın modeli çekme
kuvveti, o noktanın artığının bir fonksiyonu, ve Huber bu fonksiyonu tavanlıyor.**

Bir noktanın fit'e uyguladığı kuvvet $\left|\partial L / \partial \hat{y}\right|$:

| loss | tek noktanın çekiş kuvveti | artık 100 birim olursa |
|---|---|---|
| MSE | $2\vert u \vert$ — sınırsız büyür | 200 |
| MAE | $1$ — sabit | 1 |
| Huber | $\min(\vert u \vert, \delta)$ — $\delta$'da doyar | $\delta$ |

MSE'de tek bir uzak nokta, yakın yüzlerce noktanın toplamından daha büyük bir kuvvet
uygulayabilir. Huber'de uygulayamaz: $|u| > \delta$ olduğu anda katkısı $\delta$'da sabitlenir,
nokta ne kadar uzaklaşırsa uzaklaşsın kuvvet artmaz.

### Pizza verisiyle deney

30 noktalı veri, en sağdaki nokta (kaldıraç etkisi en yüksek olan) gerçek yerinden 75 pizza
yukarı taşınıyor. Her loss kendi optimumuna fit ediliyor, sonra o noktanın toplam gradient
çekişindeki payı ölçülüyor.

> **"Pay" burada nasıl tanımlı** *(2026-09-20'de netleştirildi — belirsizdi)*: her noktanın
> **tahmine göre** türevinin mutlak değeri, $\left|\partial L/\partial \hat{y}_i\right|$,
> toplamlarına oranlanıyor. Yani sorulan şey *"bu nokta modele ne kadar kuvvet uyguluyor"*.
>
> **Eğime göre** türev ($\partial L/\partial w$, yani $x_i$ ile ağırlıklı) alınsaydı sayılar
> farklı çıkardı — aynı deneyde MSE %42.0, MAE %7.1, Huber %11.2. İkisi de geçerli bir soru ama
> farklı sorular: birincisi *"ne kadar itiyor"*, ikincisi *"eğimi ne kadar çeviriyor"*. Aykırı
> nokta veri uzayının kenarında olduğu için ($x$ en büyük) ikinci ölçüde payı daha yüksek.
>
> Aşağıdaki tablo birinci tanıma göre. Eşit dağılsa her nokta $1/30 = 3.3\%$ olurdu.

![Tek bir aykırı değerin üç loss'taki çekiş payı](/yz50/huber-aykiri-deger-etkisi.gif)

| loss | eğim $w$ | tek noktanın payı |
|---|---|---|
| MSE | 1.695 | **31.2%** |
| MAE | 1.222 | 3.7% |
| Huber ($\delta=5$) | 1.165 | 6.1% |

Temiz veride MSE eğimi 1.081. Yani **tek bir nokta MSE'nin eğimini %57 kaydırıyor**, çünkü
toplam çekişin üçte birini tek başına üretiyor. MAE ve Huber'in eğimleri neredeyse hiç
oynamıyor.

### Bonus: MAE'nin payı neden tam 3.3% değil de 3.7%

Çünkü MAE fit'inde **3 noktanın artığı tam olarak sıfır** — $\operatorname{sign}(0) = 0$
olduğundan bu üçü toplama hiç katılmıyor, pay $1/27 = 3.70\%$ çıkıyor.

*(Bu, çözümün **tam** olmasına bağlı. Sayısal bir optimizer ile fit edersen artıklar sıfıra çok
yakın ama tam sıfır olmaz, $\operatorname{sign}$ yine $\pm 1$ verir ve pay $1/30 = 3.33\%$ çıkar.
İkisi arasındaki fark bir hata değil, "tam sıfır" ile "sıfıra yakın" arasındaki fark — ve MAE'nin
köşesinin tam da bu ayrımda yaşadığının bir başka göstergesi.)*

Bu bir tuhaflık değil, birinci bölümün doğrudan kanıtı: MAE'nin optimumu **köşenin üstünde
oturuyor.** İki parametreli bir doğru için MAE minimumu her zaman iki veri noktasından geçer
(burada üç, çünkü bir hizalanma var). Yani derivativesiz nokta kaçınılan bir istisna değil, varış
noktasının ta kendisi.

Karşılaştır: MSE'nin çözümü ortalama, MAE'ninki medyan — bkz.
[16](/posts/yz50-16-loss-nll-cross-entropy/), "hangi loss neye yakınsar". Medyanın bir veri
noktası olması ile artığın sıfır olması aynı olgunun iki söyleniş biçimi.

---

## 3. Huber'in tanımı ve $\delta$'nın rolü

**Bu bölüm neden var:** $\delta$ bir hyperparameter ve "deneyerek bul" demek yetersiz. Hangi birimde olduğunu ve neyi böldüğünü bilmek gerekiyor, yoksa seçim kör kalıyor.

$$L_\delta(u) = \begin{cases} \tfrac{1}{2}u^2 & |u| \le \delta \\[4pt] \delta\left(|u| - \tfrac{1}{2}\delta\right) & |u| > \delta \end{cases}$$

İki parça $\delta$'da hem **değerde** hem **derivativede** eşleşecek şekilde ayarlanmış:

- değer: $\tfrac{1}{2}\delta^2$ ve $\delta(\delta - \tfrac{1}{2}\delta) = \tfrac{1}{2}\delta^2$ ✓
- derivative: $\delta$ ve $\delta \cdot \operatorname{sign}(u) = \delta$ ✓

Dış parçadaki $-\tfrac{1}{2}\delta$ terimi tam olarak bunun için var; olmasaydı $\delta$'da
fonksiyon sıçrardı.

![Üç loss ve derivative'leri](/yz50/huber-uc-loss.png)

**Huber $C^1$'dir ama $C^2$ değildir** — second derivative içeride 1, dışarıda 0, $\delta$'da
sıçrıyor. Gradient descent için sorun değil; Newton tipi yöntemler kullanacaksan bil.

### $\delta$ ne anlama geliyor

$\delta$ **"bu ölçeğin altındaki sapma normal gürültü, üstündeki şüpheli"** eşiği. Birimi
artığın birimi — pizza örneğinde $\delta = 5$ demek "5 pizzalık sapma normaldir" demek.

Seçimi veriye bağlı, yaygın pratik artıkların ölçeğinden türetmek: MAD (median absolute
deviation) üzerinden $\delta \approx 1.35 \cdot \text{MAD}$ gibi. Sabit bir doğru değeri yok,
çünkü "aykırı" tanımı probleme ait bir karar.

İki uç:
- $\delta \to \infty$ → her yer kuadratik → **MSE**
- $\delta \to 0$ → her yer lineer → **MAE** (ölçeklenmiş)

Yani Huber iki loss arasında sürekli bir geçiş; $\delta$ o eksendeki konum.

PyTorch'ta `nn.HuberLoss(delta=...)` ve `nn.SmoothL1Loss(beta=...)` — ikisi $\delta$ ölçeğinde
bir çarpan farkıyla aynı fonksiyon, dokümandaki tanımı kontrol et.

---

## 4. Aynı fikrin başka yerlerdeki halleri

**Bu bölüm neden var:** "Etkiyi tavanla" fikri Huber'e özgü değil — gradient clipping, trust region, ve robust istatistiğin tamamı aynı fikrin farklı kılıkları. Deseni bir kez görünce diğerlerini tanıyorsun.

- **ReLU'nun 0'ı** birebir aynı durum: sürekli, köşeli, altderivative kümesi $[0,1]$, framework
  keyfi bir eleman seçiyor. Bkz. [13](/posts/yz50-13-aktivasyon-fonksiyonlari/) — ReLU'nun
  derivative'inin "eğri değil basamak fonksiyonu" olması tam olarak bu süreksizlik.
- **Gradient clipping** Huber'in tavanlama fikrinin parametre uzayındaki karşılığı: orada tek
  bir noktanın, burada tek bir adımın etkisi sınırlanıyor. İkisi de "büyük sinyal domine
  etmesin" diyor.
- **Robust istatistik** bu ailenin genel adı; Huber loss ismini Peter Huber'in M-estimator
  çalışmasından alıyor. MAE ve Huber aynı ailenin iki üyesi.

---

## Özet

1. Köşede derivative **yok**; seçtiğimiz değer bir sözleşme, matematiksel bir onarım değil.
   Fonksiyon $C^0$ ama $C^1$ değil.
2. Bunun pratik bedeli köşede değil, köşenin komşuluğunda: gradient küçülmediği için sabit lr
   ile minimumun etrafında salınım.
3. Huber aykırı değeri sınıflandırmıyor, **etkisini $\delta$'da tavanlıyor** — uzaklaştıkça
   çekiş artmıyor.
4. $\delta$ hem "pürüzsüzlük yarıçapı" hem "gürültü eşiği"; MSE ile MAE arasındaki eksende
   konum belirliyor.

## İlgili notlar

- [16 · Loss, NLL ve Cross-Entropy](/posts/yz50-16-loss-nll-cross-entropy/): hangi loss neye
  yakınsar (MSE → ortalama, MAE → medyan, cross-entropy → dağılımın tamamı)
- [15 · Aktivasyon + Loss Eşleşmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/): loss
  seçiminin arkasındaki GLM teorisi
- [13 · Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/): ReLU'nun aynı
  köşesi

Görseller [img/make_huber_figures.py](/yz50/make_huber_figures.py) ile üretiliyor.
