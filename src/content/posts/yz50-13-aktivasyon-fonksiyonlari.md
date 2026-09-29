---
title: "Hafta 2 — Aktivasyon Fonksiyonları: Sigmoid vs Tanh vs ReLU"
published: 2026-09-29
description: "Sigmoid, tanh ve ReLU'nun türevleri, doygunluk davranışı ve 'tanh her zaman daha güçlü' iddiasının nerede bozulduğu (x≈1.663)."
tags:
  - YZ50
  - Aktivasyon
category: YZ50
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

Kaynak: [GeeksforGeeks — Tanh vs. Sigmoid vs. ReLU](https://www.geeksforgeeks.org/deep-learning/tanh-vs-sigmoid-vs-relu/) (kullanıcı yapıştırdı, sadeleştirilmiş özeti aşağıda).

---

## Neden Var

Ağırlıklı toplam + bias'tan sonra bir **nonlineer** fonksiyon uygulanmazsa, ağ kaç layer'lı olursa olsun matematiksel olarak **tek bir doğrusal fonksiyona** indirgenir (doğrusal layer'ların üst üste binmesi yine doğrusaldır) — gerçek dünyanın karmaşık, nonlineer ilişkilerini hiç öğrenemez. Aktivasyon fonksiyonu bu doğrusallığı kırıyor.

![Sigmoid, Tanh ve ReLU eğrileri](/yz50/aktivasyon-fonksiyonlari.png)

## Sigmoid

**Bu bölüm neden var:** Tarihsel olarak ilk kullanılan aktivasyon ve hâlâ ikili sınıflandırmanın çıkışında duruyor. Ama hidden layer'larda neden terk edildiğini anlamak için önce ne yaptığını görmek gerekiyor.

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

- **Range:** 0 ile 1 arası
- **Ne zaman kullanılır:** binary classification'da **output layer**'da (çıktının probability olarak yorumlanması gerektiği yerler), logistic regression
- **Sorun:** uçlara (büyük +/- değerlere) gittikçe eğim neredeyse 0'a düşüyor → **vanishing gradient** (bu terimi C'deki GC debugging analojisinden zaten biliyorsun). Ayrıca çıktı hep pozitif (zero-centered değil), bu da weight update'leri verimsizleştiriyor.

## Tanh

**Bu bölüm neden var:** Sigmoid'in kaydırılmış/ölçeklenmiş hali — ama o küçük fark (zero-centered olması) hidden layer'da tercih edilmesinin sebebi. Farkın nerede biteceği de burada çıkıyor.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **Range:** -1 ile +1 arası, **zero-centered** (sigmoid'den farkı ve avantajı burada)
- **Ne zaman kullanılır:** **hidden layer**'larda sigmoid yerine — zero-centered olduğu için gradient akışı daha sağlıklı, öğrenme genelde daha hızlı yakınsıyor
- **Sorun:** yine de uçlarda saturate oluyor, vanishing gradient sigmoid kadar şiddetli olmasa da hâlâ var

**Vanishing, tek tarafta değil, iki uçta da simetrik.** $\tanh'(x)=1-\tanh^2(x)$'e bak: $x \to +\infty$'da $\tanh \to +1$, dolayısıyla $\tanh' \to 1-1^2 = 0$. $x \to -\infty$'da $\tanh \to -1$, dolayısıyla $\tanh' \to 1-(-1)^2 = 0$ — kare alındığı için işaret kayboluyor, **-1'de de aynı şekilde sıfıra gidiyor.** Sigmoid'te de aynı simetri var ($\sigma' \to 0$ hem $x \to +\infty$'da hem $x \to -\infty$'da, yani hem 1'e hem 0'a yaklaşırken). İkisi de $x=0$'da tepe yapan, iki uca doğru simetrik alçalan birer "tümsek" — sadece tümseğin yüksekliği farklı (1 vs 0.25).

**Bu, Hafta 2'de `Value`/`Neuron` sınıflarında kullandığın fonksiyon** — Karpathy'nin micrograd'ı da sigmoid değil tanh tercih ediyor, tam bu yüzden (hidden layer'larda zero-centered olması işe yarıyor).

### Neden Sigmoid Kadar Şiddetli Değil — Kesin İlişki

**Önce: $x=0$ neden keyfi bir seçim değil.** $\sigma'(x)$ ve $\tanh'(x)$ birer fonksiyon, $x$'e göre değişiyor — "en yüksek noktası nerede" sorusunun cevabı tahmin değil, **kritik nokta hesabı**: derivative'in derivative'ini al, sıfıra eşitle.

Sigmoid için: $\sigma''(x) = \sigma'(x)\,(1-2\sigma(x))$. $\sigma'(x)$ hiç tam sıfır olmadığı için (sigmoid kesin artan), tek çözüm $1-2\sigma(x)=0 \Rightarrow \sigma(x)=\tfrac{1}{2} \Rightarrow x=0$. $\sigma'(x)$ uçlarda ($x \to \pm\infty$) 0'a gittiği için bu tek kritik nokta bir **maksimum.** Tanh için de aynı mantıkla: $\tanh''(x) = -2\tanh(x)\,(1-\tanh^2(x)) = 0 \Rightarrow \tanh(x)=0 \Rightarrow x=0$. Yani $x=0$, ikisinin de maksimumunun **ispatlanmış** yeri — keyfi seçilmedi.

**$x=0$ bir eğitim hedefi değil.** Ağı eğitirken "$z$'yi 0'a yaklaştır" diye bir amaç yok — z, veri ve öğrenilen weight'lera göre ne çıkarsa o. $x=0$'ın önemi sadece şu: **eğer** bir nöronun $z$'si o an 0'a yakınsa, o nöron için gradient sinyali mümkün olan en güçlü halinde. Bu bir hedef değil, iki fonksiyonun en iyi durumda ne kadar iyi olabileceğini kıyaslamak için kullanılan bir **referans noktası.**

**Değerler neden tam 0.25 ve tam 1 — yaklaşık değil:** $\tanh(0) = \dfrac{e^0 - e^{-0}}{e^0 + e^{-0}} = \dfrac{0}{2} = 0$ (tanımdan kesin), dolayısıyla $\tanh'(0) = 1-0^2 = 1$. Sigmoid için $\sigma(0) = \dfrac{1}{1+e^0} = \tfrac{1}{2}$, dolayısıyla $\sigma'(0) = 0.5 \times 0.5 = 0.25$. Yuvarlama yok, cebirsel zorunluluk.

Derivativelerin en yüksek noktaları ($x=0$'da): $\sigma'(0) = 0.25$, $\tanh'(0) = 1$. Tesadüf değil — cebirsel bir özdeşlik var: $\tanh(x) = 2\sigma(2x) - 1$. Bunu derivative'lersen:

$$\tanh'(x) = 4\,\sigma'(2x)$$

**Semboller:** $\tanh'(x)$ = tanh'in derivative'i · $\sigma'(2x)$ = sigmoid'in derivative'i, ama $2x$ noktasında değerlendirilmiş (girdi ölçeklenmiş) · 4 = sabit çarpan, $\tanh(x)=2\sigma(2x)-1$ özdeşliğinden chain ruleyla çıkıyor (dış derivative 2, iç derivative $2x$'in derivative'i 2, $2 \times 2 = 4$).

Yani **peak noktasında** ($x=0$) tanh'in derivative'i sigmoid'inkinden **tam 4 kat büyük** (bu oranın yalnızca orada geçerli olduğu aşağıda). Çok layer'lı bir ağda bu çarpanlar zincirleme çarpılıyor (backprop) — 10 layer'da kabaca $0.25^{10} \approx 9 \times 10^{-7}$ (sigmoid, neredeyse sıfır) ile $1^{10} = 1$ (tanh, en iyi durumda hiç sönmüyor) arasındaki fark. **Vanishing gradient'in "orta" vs "şiddetli" olmasının kesin sayısal kaynağı bu 4x'lik fark.**

**Bu bize tam olarak ne gösteriyor:** backprop'ta bu terimler layer layer çarpılıyor, ve her layer'da $z$ ne olursa olsun o layer'ın çarpanı **kendi maksimumunu asla geçemez.** Yani bu bir **üst sınır** karşılaştırması: **en iyi senaryoda bile** (her nöron tam $z=0$'da) sigmoid layer başına en fazla $\times 0.25$ verebilir, tanh en fazla $\times 1$. Sonuç sadece "tanh genelde daha iyi" değil — "sigmoid en iyi durumda bile yapısal olarak daha kötü, kaçışı yok."

**Neden başka bir $x$ değil — oran her yerde 4 değil:** $\tanh'(x)=4\sigma'(2x)$ özdeşliği her $x$'te geçerli, ama $\sigma'(x)$ ile $\tanh'(x)$'i **aynı $x$'te** karşılaştırırsan oran sabit kalmıyor. Örnek, $x=1$'de: $\sigma'(1) \approx 0.1966$, $\tanh'(1) \approx 0.4200$, oran $\approx 2.14$ (4 değil). Temiz "4 kat" iddiası **sadece** peak noktalarında ($x=0$, ikisi için de) çıkıyor — $x=0$'ı seçmemin sebebi tam bu: hem matematiksel olarak zorunlu maksimum, hem genelleştirilebilir tek nokta.

**Ve oran yalnızca küçülmüyor, $x \approx 1.663$'ten sonra 1'in de altına iniyor** — yani orada
tanh, sigmoid'den *zayıf*. Bunun tam tablosu, grafiği ve kendi `noron()` kodundan gerçek sayılarla
sonucu aşağıda, **"Kesişim Noktası"** bölümünde.

Oran $x=1$'de 2.14'e düşüyor dedik — ama bu küçülme orada durmuyor, bir noktadan sonra **tamamen tersine dönüyor:**

$$\tanh'(x) = \sigma'(x) \quad \text{kesişimi } x \approx 1.663 \text{'te oluyor}$$

| $x$ | $\tanh'(x)$ | $\sigma'(x)$ | kim büyük |
|---|---|---|---|
| 0.0 | 1.00000 | 0.25000 | tanh |
| 1.0 | 0.41997 | 0.19661 | tanh |
| 1.6 | 0.15053 | 0.13976 | tanh |
| 1.663 | $\approx 0.1310$ | $\approx 0.1310$ | **kesişim** |
| 1.7 | 0.12501 | 0.13061 | sigmoid |
| 2.0 | 0.07065 | 0.10499 | sigmoid |
| 3.0 | 0.00987 | 0.04518 | sigmoid |

![tanh' ve sigma' eğrileri, tepe noktaları ve kesişim](/yz50/tanh-sigmoid-derivative-kesisim.png)

$x \approx 1.663$'ten sonra **$\sigma'(x)$, $\tanh'(x)$'ten büyük** — yani tanh, sigmoid'den daha hızlı doyuyor. Sebebi aynı özdeşlik ($\tanh(x)=2\sigma(2x)-1$): tanh, sigmoid'in girdi ekseninde 2 kat sıkıştırılmış/dikleştirilmiş hali. Bu dikleşme $x=0$ civarında daha büyük eğim veriyor (avantaj) — ama tam o yüzden eğri daha erken (daha küçük $|x|$'te) yassılaşmaya başlıyor (dezavantaj). Biri diğerinin bedeli, aynı özdeşlikten çıkan iki yüz.

**Somut örnek — Hafta 2'deki `noron()` fonksiyonundan gerçek sayılar:** `w1=-3.0` gibi büyükçe bir weightla `n=3.0` gibi bir pre-activation'a ulaşınca (kesişim noktasının epey ötesi), tanh sigmoid'e göre **dezavantajlı** duruma düşüyor:

| | tanh | sigmoid | oran |
|---|---|---|---|
| `n.grad` ($\partial \text{çıktı}/\partial n$) | 0.009866 | 0.045177 | sigmoid **4.58 kat** büyük |
| x1.grad | -0.029598 | -0.135530 | aynı oran |
| w1.grad | 0.019732 | 0.090353 | aynı oran |

Zincirdeki her leaf (x1, x2, w1, w2, b) tam olarak aynı oranla etkileniyor — çünkü iki hesaplama grafiğinin tek farkı `n`'den sonraki son adım (aktivasyon fonksiyonu), `n`'ye kadar olan her şey (x1w1, x2w2, x1w1x2w2) birebir aynı. Chain rule'da tüm fark tek noktada toplanıyor ve oradan aşağı sabit bir çarpan gibi yayılıyor.

**Sonuç — "tanh daha az vanishing gradient yaşar" iddiası koşullu:** bu, **pre-activation'ın 0 civarında olduğu** varsayımına dayanıyor — düzgün weight initialization (Xavier/He) ile taze bir ağda makul bir varsayım. Ama weight'ler büyükse (ya da normalization yoksa), pre-activation 0'dan uzaklaşabilir, ve tanh'in "avantajı" yukarıdaki örnekteki gibi tam tersine dönebilir. Bu, weight initialization'ın ve batchnorm/layernorm gibi tekniklerin neden önemli olduğunun somut bir gerekçesi: pre-activation'ları eğrinin "sağlıklı" (0'a yakın) bölgesinde tutmak.

### Zero-Centered'ın Somut Etkisi — Küçük Örnek

Weight gradient'i formülü (bkz. [12](/posts/yz50-12-backpropagation-calculus/)):

$$\frac{\partial C}{\partial w_{jk}} = a_k \cdot \sigma'(z_j) \cdot (\text{üstten gelen terim})$$

Bir nöronun **tüm** gelen weight'leri için $\sigma'(z_j)$ ve üstten gelen terim **aynı** — sadece $a_k$ (önceki layer'ın aktivasyonu) değişiyor.

**Sigmoid'de** ($a_k$ hep pozitif — örn. $a_1=0.3$, $a_2=0.7$): iki weight'in gradient'i **aynı işaretli** çıkıyor (mesela 1.5 ve 3.5, ikisi de +). Gradient descent'te ikisi de **aynı yönde** hareket etmek zorunda — biri artarken diğeri aynı adımda azalamıyor.

**Tanh'te** ($a_k$ hem + hem - olabilir — örn. $a_1=-0.3$, $a_2=0.7$): gradient'ler **farklı işaretli** çıkabiliyor (-1.5 ve +3.5). $w_{j1}$ artarken $w_{j2}$ aynı adımda azalabiliyor — sigmoid'in "hepsi birlikte hareket etmeli" kısıtı kalkıyor, güncelleme daha doğrudan/verimli oluyor.

## ReLU

**Bu bölüm neden var:** İlk ikisinin ortak sorunu doyma. ReLU o sorunu pozitif tarafta tamamen ortadan kaldırıyor — ve karşılığında yeni bir sorun getiriyor (ölü nöronlar).

$$f(x) = \max(0, x)$$

- **Range:** 0'dan sonsuza — negatifse 0, pozitifse kendisi
- **Ne zaman kullanılır:** derin ağlarda (CNN, büyük MLP (Multi-Layer Perceptron)'ler) günümüzde **varsayılan tercih** — hesaplaması ucuz (sadece bir karşılaştırma, exponential yok), pozitif tarafta vanishing gradient hiç yaşanmıyor
- **Sorun:** "dying ReLU" — bir nöron sürekli negatif girdi alırsa gradient'i tamamen 0'a kilitlenir, bir daha hiç öğrenmez (kalıcı ölü nöron)

**ReLU'nun derivative'i eğri değil, basamak fonksiyonu — sigmoid/tanh'tan tamamen farklı bir hikaye:**

$$f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

x>0 için derivative **her zaman tam 1** — x=1 olsun, x=1.000.000 olsun fark etmez, hiç küçülmüyor. Yani ReLU'nun çıktısı sonsuza gidebilse de, **derivative'i büyüklüğü hiç umursamıyor** — sigmoid/tanh'taki "uzaklaştıkça yavaşça sıfıra sönme" hikayesi ReLU'nun pozitif tarafında **hiç yok.**

Ama negatif tarafta farklı bir sorun var: sigmoid/tanh gibi "yavaşça küçülme" değil, **anlık ve kalıcı sıfır** — z negatife düşerse derivative tam 0, "biraz küçük" değil, **hiç sinyal yok.** Yani ReLU vanishing gradient'i **büyüklükle** değil (sigmoid/tanh'ın simetrik tümseği gibi), **işaretle** (pozitif mi negatif mi, ikili/binary) yaşıyor — kademeli değil, anlık.

*(Bonus: bu kalıcı ölüm sorununu hafifletmek için **Leaky ReLU** var — negatif tarafta tam 0 yerine çok küçük bir eğim (0.01 gibi) veriyor, nöron tamamen ölmüyor. Faz 4'te karşına çıkabilir.)*

## Karşılaştırma

**Bu bölüm neden var:** Üçünü tek tabloda görmeden "hangisini seçeyim" sorusuna cevap veremezsin.

| | Sigmoid | Tanh | ReLU |
|---|---|---|---|
| Range | 0 – 1 | -1 – +1 | 0 – ∞ |
| Zero-centered | Hayır | Evet | Hayır |
| Vanishing gradient | Şiddetli | Orta | Yok (pozitif tarafta) |
| Maliyet | Pahalı (exp) | Pahalı (exp) | Çok ucuz (max) |
| Tipik kullanım | Output layer (binary classification) | Hidden layer (küçük/orta ağlar) | Hidden layer (derin ağlar, CNN) |

## Dönüm Noktası ile Cost Fonksiyonu Arasındaki Bağlantı

**Bu bölüm neden var:** Aktivasyonun derivative'i tek başına bir eğri özelliği. Asıl önemli olan o eğrinin **weight güncellemesine** nasıl taşındığı — chain rule üzerinden, ve bu vanishing gradient'in tam mekanizması.

$\sigma'(z)$ ve $\tanh'(z)$, cost fonksiyonunun kendisi değil — chain rulendaki (bkz. [12](/posts/yz50-12-backpropagation-calculus/)) üç çarpandan biri:

$$\frac{\partial C}{\partial w} = a^{(L-1)} \cdot \sigma'(z) \cdot 2(a-y)$$

Çarpım olduğu için: çarpanlardan biri sıfıra yakınsa, diğerleri ne kadar büyük olursa olsun **sonuç sıfıra yakın olmak zorunda.**

**Sayısal örnek — sigmoid, iki senaryo:** $a^{(L-1)}=0.5$ ve hata $2(a-y)=10$ (ağ çok yanlış) sabit tutuluyor.

- $z \approx 0$ (dönüm noktasında, en iyi durum): $\sigma'(0)=0.25 \Rightarrow \partial C/\partial w \approx 0.5 \times 0.25 \times 10 = \mathbf{1.25}$
- $z = 10$ (doymuş): $\sigma'(10) \approx 0.0000454 \Rightarrow \partial C/\partial w \approx 0.5 \times 0.0000454 \times 10 \approx \mathbf{0.000227}$, neredeyse sıfır

![Doyma bölgesinde gradient'in ölmesi](/yz50/saturation-gradient.png)

**İkisinde de hata (10) aynı** — ağ ikisinde de eşit yanlış. Ama doymuş durumda $\sigma'(z)$'nin sıfıra yakın olması, hatanın büyüklüğünü önemsizleştiriyor — vanishing gradient tam olarak bu: aktivasyonun kendi dönüm noktasından uzakta olması (doyması), cost'un o weight için üretebileceği gradient'i bloke ediyor, hata ne kadar büyük olursa olsun.

**Tanh ile karşılaştırma — dikkatli okunmalı:** sigmoid'in **en iyi durumu** (1.25), tanh'ın **en iyi durumundan** (aynı hesapla: $0.5 \times 1 \times 10 = 5$) 4 kat küçük. Ama bu karşılaştırma **ikisinin de $z=0$'da olduğu** varsayımına dayanıyor.

> **Düzeltme (2026-09-20):** Bu paragrafın önceki hali "tanh her zaman en az 4 kat daha güçlü
> sinyal veriyor" diyordu. **Yanlıştı.** Yukarıdaki tabloda görüldüğü gibi oran $z$ ile düşüyor
> ve $z \gtrsim 1.66$'dan sonra **1'in altına iniyor** — orada tanh sigmoid'den *zayıf*.
> $z=2$'de aynı hesap: tanh $0.5 \times 0.0707 \times 10 = 0.354$, sigmoid $0.5 \times 0.105
> \times 10 = 0.525$.

Doğru okunuşu: **tanh'ın avantajı doyma bölgesine girilmediği sürece var.** Girilirse ikisi de
ölüyor ve tanh — argümanı iki katına çıkardığı için — daha hızlı ölüyor. Bu yüzden tanh'ı
tercih etmek tek başına yetmiyor; $z$'yi 0 civarında tutan bir init/normalizasyon şart.

**Özet:** dönüm noktası aktivasyon fonksiyonunun kendi özelliği; cost'un gradient'i bu özelliği chain ruleyla çarparak devralıyor. Aktivasyon dönüm noktasına yakınsa cost gradient'i güçlü akar, doymuşsa cost gradient'i hata ne kadar büyük olursa olsun boğulur.

## Girdi Ölçeği ve İlk Layer'ın Özel Riski

**Bu bölüm neden var:** İlk layer'ın girdisi ham veri, yani ölçeği senin kontrolünde değil. Bu, [01](/posts/yz50-01-lineer-regresyon/)'deki condition number sorununun aktivasyon tarafındaki karşılığı — ve [14](/posts/yz50-14-kaiming-init-ve-batchnorm/)'ün varlık sebebi.

İlk layer'ın girdisi (x) ham veri — ölçeği tamamen veriye bağlı (piksel 0-255, gelir on binlerce olabilir), senin kontrolünde değil. $z = w \cdot x + b$ hesaplanırken $x$ zaten büyükse, $w$ küçük bile olsa $z$ kolayca büyür, nöron doyar — eğitim daha başlamadan.

**Sonraki (hidden) layer'lar farklı — girdileri zaten sınırlı.** İkinci layer'ın girdisi, bir önceki layer'ın aktivasyon çıktısı — sigmoid'den geçmişse (0,1), tanh'tan geçmişse (-1,1) aralığında, ham veri gibi sınırsız değil. Bu yüzden "girdi ölçeği çok büyük" problemi özellikle **ilk layer'a özgü.**

**Somut örnek ($w=0.1$ sabit, sadece $x$'in ölçeği değişiyor):**

| $x$ (ham) | $z$ | $\tanh'(z)$ |
|---|---|---|
| 1 | 0.10 | 0.990066 |
| 50 | 5.00 | 0.000182 |
| 1000 | 100.00 | 0.000000 |

![Ham girdi ölçeği ile derivative'in sönmesi](/yz50/girdi-olcegi.png)

Aynı $x$'i önce 0-1 aralığına normalize edersen ($x=1000 \Rightarrow 1.0$), $z$ sadece 0.1'e düşüyor, derivative sapasağlam (0.99) kalıyor.

**Bu, "input normalization/standardization" adı verilen, ML'de standart bir ön işlem adımının gerekçesi** — ham veriyi doğrudan ilk layer'a vermek yerine önce ölçeklendirmek (ortalama 0/std 1, ya da 0-1 aralığı), ilk layer'ın doyma bölgesine düşmesini önlüyor. Sonraki layer'lar için benzer bir rolü **weight initialization** (Xavier/He) ve **batch/layer normalization** üstleniyor.

## Çok Layerlı Zincir — Çıktı Layer'ının Seçimi İlk Layera Nasıl Ulaşıyor

İki layer'lı bir ağ düşün: girdi → gizli layer (tanh) → çıktı layer'ı (sigmoid ya da tanh).

$$z_1 = w_1 x + b_1 \qquad a_1 = \tanh(z_1)$$

$$z_2 = w_2 a_1 + b_2 \qquad a_2 = \phi_{\text{çıktı}}(z_2)$$

İlk layer'daki $w_1$'e ulaşan tam gradient, beş çarpanın çarpımı:

$$\frac{\partial C}{\partial w_1} = \frac{\partial C}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

$\partial a_2/\partial z_2$ (çıktı layer'ının kendi derivative'i) **ve** $\partial a_1/\partial z_1$ (gizli layer'ın kendi derivative'i) — ikisi de aynı zincirde, ikisi de birer çarpan. Çıktı layer'ının hangi aktivasyonu kullandığı, en baştaki $w_1$'e ulaşan gradient'i da etkiliyor.

**Somut sayılarla ($x=2$, $w_1=3$ — bilerek gizli layer'ı doymuş bir bölgede tutuyoruz, $z_1=6$, $\tanh'(z_1) \approx 0.0000246$):**

| Çıktı layer'ı | $\partial a_2/\partial z_2$ | $w_1$'e ulaşan tam gradient |
|---|---|---|
| sigmoid | 0.235004 | 0.0000058 |
| tanh | 0.786452 | 0.0000193 |

Sadece çıktı layer'ını değiştirerek, w1'e ulaşan gradient ~3.3 kat değişti — çıktı layer'ının seçimi gerçekten en baştaki girdinin etkisini değiştiriyor.

**Ama bir sınır var:** gizli layer zaten çok doymuşsa (buradaki gibi), çıktı layer'ının seçimi ne olursa olsun sonuç hâlâ çok küçük — zincirdeki her halka çarpana giriyor, biri feci küçükse diğerleri onu kurtaramıyor. Önemli olan tek bir layer'ın seçimi değil, **zincirdeki her layer'ın toplamda** ne kadar sağlıklı olduğu.

## Hangi Layerda Hangi Aktivasyon — Pratik Kural

**Bu bölüm neden var:** Teorinin sonunda bir karar vermek gerekiyor. Bu bölüm kuralı veriyor ama **gerekçesiyle**, çünkü gerekçesiz kural bir sonraki farklı durumda işe yaramıyor.

**Çıktı layer'ı — bu bir seçim değil, görev tipi belirliyor:**
- İkili sınıflandırma (evet/hayır) → **sigmoid** + binary cross-entropy ([15](/posts/yz50-15-aktivasyon-loss-eslesmesi/)'deki canonical pairing)
- Çok sınıflı sınıflandırma → **softmax** + categorical cross-entropy
- Regresyon (sınırsız, sürekli sayı — örn. tur zamanı tahmini) → **doğrusal/aktivasyon yok** — sigmoid/tanh'ın sınırlı aralığı, sınırsız bir hedefi tahmin etmeye engel olur

**Gizli layer'lar — tarihsel olarak tanh, modern pratikte ReLU:**
- tanh, zero-centered olduğu için sigmoid'e tercih edilirdi (yukarıdaki tablo)
- Modern derin ağlarda varsayılan artık **ReLU** — yukarıdaki çok-layer'lı-zincir sorununu çözüyor: pozitif tarafta derivative hep tam 1, hiç doymuyor, çok layer'lı bir ağda çarpanlar sıfıra gitmiyor
- İstisna: LSTM (Long Short-Term Memory)/GRU (Gated Recurrent Unit) gate'leri özellikle sigmoid kullanır (çıktının (0,1) aralığında olması "ne kadarını geçireyim" anlamına geliyor) — mimari bir gereklilik, genel kuralın istisnası

## Girdi (z) ile Çıktı (a) Karıştırılmamalı (2026-08-31)

**$z$ (aktivasyona giren ham değer) sınırsız** — herhangi bir reel sayı olabilir. **$a=\phi(z)$ (çıktı) sınırlı** — sigmoid için $(0,1)$, tanh için $(-1,1)$. Kesişim noktası olan $z \approx 1.663$, sigmoid/tanh'ın **çıktısı** değil, **girdisidir** — $z=1.663$'te bile $\sigma(z) \approx 0.84$, hâlâ (0,1) içinde, kural bozulmuyor.

**$\sigma'(z)$ ve $\tanh'(z)$ ikisi de çift (even) fonksiyon** — yani $\sigma'(-z)=\sigma'(z)$ ve $\tanh'(-z)=\tanh'(z)$. Bu yüzden kesişim tek taraflı değil, simetrik:

| $z$ | $\tanh'(z)$ | $\sigma'(z)$ | kim büyük |
|---|---|---|---|
| -3.0 | 0.0099 | 0.0452 | sigmoid |
| -1.663 | 0.1339 | 0.1340 | kesişim |
| 0.0 | 1.0000 | 0.2500 | tanh |
| 1.663 | 0.1339 | 0.1340 | kesişim |
| 3.0 | 0.0099 | 0.0452 | sigmoid |

$z$, $[-1.663,\ 1.663]$ aralığındaysa (0-1 aralığı bunun tamamen içinde) tanh kazanıyor; $|z| > 1.663$ olunca (her iki yönde de) sigmoid öne geçiyor.

## Hidden Layer'da z Neden Hâlâ Büyüyebilir — Çoklu Nöron Toplamı

Bir önceki layer'dan gelen her tekil çıktı ($a_k$) sınırlı olsa da, bir sonraki layer'daki nöronun $z$'si **hepsinin ağırlıklı toplamı** — ve toplam sınırsız büyüyebilir (bkz. [12](/posts/yz50-12-backpropagation-calculus/)'daki $z_j^{(L)}=\sum_k w_{jk}^{(L)} a_k^{(L-1)}+b_j^{(L)}$ formülü):

**Somut örnek:** 50 nöronluk bir önceki layer, her $a_k$ gerçekten $(-1,1)$ arasında sınırlı, her weight da makul (0.5-1.5, hiç "büyük" değil) — ama toplam $z=-3.567$ çıkıyor, kesişim noktasının (1.663) 2 katından fazla. Sınırlı girdiler, toplamın da sınırlı kalacağını garanti etmiyor — kaç terim topladığın (fan-in) önemli.

Bu, weight initialization'ın (Xavier/He) neden fan-in'e göre weight'i **küçülterek** başlattığının gerekçesi — toplamayı önceden dengelemek için.

## Xavier vs He — Hangisi Neden Hangi Aktivasyona Uyuyor

**Bu bölüm neden var:** İki init şeması var ve hangisinin seçileceği aktivasyona bağlı. Sebebi tesadüfi değil — [14](/posts/yz50-14-kaiming-init-ve-batchnorm/)'teki varyans hesabından çıkıyor.

- **Xavier (Glorot):** varsayımı, aktivasyonun 0 civarında **simetrik ve yaklaşık doğrusal** olması. tanh tam bu — tek (odd) fonksiyon ($\tanh(-z)=-\tanh(z)$), ve $\tanh'(0)=1$ olduğu için $z=0$ civarında $\tanh(z) \approx z$ (neredeyse doğru orana yakın). Xavier'in varyans-koruma hesabı bu varsayıma dayanıyor — tanh için neredeyse birebir uyuyor, sigmoid için de kullanılıyor ama sigmoid tam simetrik olmadığı (çıktısı (0,1), orijinden geçmiyor) için biraz daha az kusursuz.
- **He (Kaiming):** ReLU **simetrik değil** — negatif tarafı tamamen sıfırlıyor, nöronların yaklaşık yarısı her zaman ölü. Xavier'in simetri varsayımı burada çöküyor. He, bu kaybı telafi etmek için varyans hesabına bir **2 çarpanı** ekliyor.
- **Pratikte (PyTorch'un `nn.init` modülünde birebir):** `xavier_uniform_`/`xavier_normal_` → tanh/sigmoid, `kaiming_uniform_`/`kaiming_normal_` (He'nin diğer adı) → ReLU ailesi.

*(İleride daha derinlemesine incelenecek bir konu.)*

## Basit NumPy Karşılığı

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
```

---

## Bağlantı

Hafta 2'nin `Value`/`Neuron`/`Layer`/`MLP` implementasyonunda aktivasyon olarak **tanh** kullanılıyor — yukarıdaki tabloya göre bunun sebebi hidden layer'da zero-centered olmasının gradient akışını sigmoid'e göre iyileştirmesi. Faz 4'te (PyTorch, büyük ağlar) muhtemelen ReLU ailesiyle daha çok karşılaşacaksın.

