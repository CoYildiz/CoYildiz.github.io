---
title: "Sınıflandırmaya Geçiş — Lineer Regresyonun Sınırı ve Sigmoid"
published: 2026-09-29
description: "Regresyon neden sınıflandırmada çuvallıyor: accuracy'nin optimize edilemeyeceğinin ispatı, MSE'nin elenme gerekçesi ve confusion matrix okuma."
tags:
  - YZ50
  - Sınıflandırma
  - Sigmoid
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

**Kaynak:** Educative, *Fundamentals of Machine Learning for Software Engineers*, **Bölüm 5 —
A Discern Machine**. Dersler: *Linear Regression Limitation* → *Invasion of the Sigmoids* →
*Update the Gradient* → *Classification in Action* → *Playground (Weighty Decisions)*.
Bu kurs Paolo Perrotta'nın *Programming Machine Learning* kitabının Educative sürümü.

Önceki notlar: [01-lineer-regresyon](/posts/yz50-01-lineer-regresyon/),
[02-hiperuzay](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/).

> **Durum:** *Invasion of the Sigmoids* dersi okundu ve not o dersin içeriğiyle **doğrulandı**
> (2026-09-19). *Linear Regression Limitation*, *Update the Gradient* ve *Classification in Action*
> henüz okunmadı — o bölümler hâlâ okuma yoldaşı niteliğinde, işaretli.

---

## 1. Bu bölüm nereye oturuyor

**Bu bölüm neden var:** Kitabın bu bölümü dört ders içeriyor ve her biri modelin farklı bir parçasını değiştiriyor. Hangi dersin neyi değiştirdiğini önden bilmek, okurken kaybolmayı önlüyor.

01 ve 02'de model bir **sayı** tahmin ediyordu: kaç pizza satılacak. Bu bölümde soru değişiyor:
**evet/hayır.** Çıktı artık sürekli bir miktar değil, iki etiketten biri.

Model gövdesi aynı kalıyor — `X @ w`, bias hilesiyle birlikte ([02](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/),
bölüm 2). Değişen üç şey var ve bölümün dört dersi tam olarak bunlar:

| Ders | Ne değişiyor |
|---|---|
| Linear Regression Limitation | **Problem:** neden mevcut model kırılıyor |
| Invasion of the Sigmoids | **Model:** çıktıya sigmoid ekleniyor |
| Update the Gradient | **Loss ve derivative:** MSE gidiyor, log loss geliyor |
| Classification in Action | **Sonuç:** karar sınırı ve doğruluk ölçümü |

---

## 2. Lineer regresyon sınıflandırmada neden kırılıyor *(ders henüz okunmadı)*

Üç ayrı sebep var; üçü de bağımsız olarak yeter.

**Çıktı sınırsız.** Etiket 0 veya 1, ama `X @ w` istediği değeri alabilir: −4 de çıkabilir, 17 de.
"Model %1700 emin" cümlesinin anlamı yok. Eşik koyup kesebilirsin ama bu modeli düzeltmiyor,
çıktısını gizliyor.

**Doğru taraftaki aykırı örnek karar sınırını sürüklüyor.** Bu en sinsi olanı. MSE kare hata
topluyor; sınırın çok uzağında, **doğru** sınıflandırılmış bir örnek bile büyük bir artık üretir
(çünkü tahmin 5, etiket 1 → hata 4, karesi 16). Model bu hatayı küçültmek için doğruyu ona doğru
eğer — ve bunu yaparken sınıra yakın örnekleri yanlış tarafa atar. **Zaten doğru bildiğin bir
örnek, modeli bozuyor.** Regresyonda bu davranış mantıklıydı; sınıflandırmada saçma.

![Aykırı örnek uzaklaştıkça karar sınırı kayıyor](/yz50/linreg-siniflandirmada-kirilir.gif)

Yeşil çemberli nokta hep **doğru tarafta** ve etiketi hiç değişmiyor — sadece uzaklaşıyor. Doğru
onu yakalamak için eğiliyor, kırmızı karar sınırı sağa kayıyor, ve sınıra yakın örnekler
(siyah çerçeveli) teker teker yanlış tarafa düşüyor. Model hiçbir yeni bilgi almadı; sadece zaten
doğru bildiği bir örnek daha uzağa gitti.

**0.5 eşiği keyfi.** Çıktı bir probability değilse, hangi sayıda keseceğin modele değil sana bağlı
kalıyor. Probability yorumu olmadan "bu tahminden ne kadar eminim" sorusunun cevabı yok
(bkz. [05-ag-analizi](/posts/yz50-05-ag-analizi-ve-sinirlari/) — eğitilmiş ağın gürültüye yüksek güvenle
yanlış cevap vermesi, aynı kalibrasyon sorunu).

---

## 3. Sigmoid ne yapıyor

**Bu bölüm neden var:** Bölüm 2'deki üç sorunun da tek bir çözümü var ve o çözüm sigmoid. Neyi nasıl düzelttiğini görmek gerekiyor — özellikle de üçüncü sorunu (aykırı örneğin sürüklemesi) kendiliğinden kapatmasını.

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Herhangi bir gerçel sayıyı `(0, 1)` aralığına sıkıştırıyor. Model iki layer'lı bir boru hattına
dönüşüyor:

```
z = X @ w           # eskisiyle aynı, bias w[0] içinde
tahmin = sigmoid(z) # yeni: (0,1) aralığında
```

Kazanılan üç şey:

![Sigmoid herhangi bir sayıyı 0-1 aralığına sıkıştırıyor](/yz50/sigmoid-sikistirma.gif)

Solda ham `z` sınırsız geziniyor, sağda sigmoid'den geçmiş hali `0` ile `1` arasında kalıyor.
`z` büyüdükçe çıktı `1`'e yaklaşıyor ama **hiç varmıyor** — ve `classify()` yalnızca kırmızı
eşiğin hangi tarafında olduğuna bakıyor.

- **Sınırlı çıktı.** Artık 17 çıkamaz
- **Probability gibi okunabilir çıktı.** 0.82 → "bu örnek muhtemelen 1"
- **Uzaktaki doğru örnek artık sürüklemiyor.** `z = 5` de olsa `z = 50` de olsa sigmoid ikisini de
  ~1'e bastırıyor, yani ikisinin hatası da ~0. İkinci sorun kendiliğinden kapanıyor

`sigmoid(0) = 0.5`; negatif tarafta hızla 0'a, pozitif tarafta 1'e yaklaşıyor ama **ikisine de
hiç ulaşmıyor.**

**İki isim değişikliği bu derste oluyor, ikisi de kalıcı:**

- `predict()` → **`forward()`**. Gerekçe: veriyi sistemin içinden ileri doğru geçirme işleminin adı
  **forward propagation**. Bu isim kursun geri kalanında ve YZ50'de aynen kullanılıyor —
  `mlp_explicit.py`'de yazdığın şeyin adı da bu
- Ayrı bir **`classify()`** fonksiyonu geliyor: `forward()`'ın (0,1) arası çıktısını en yakın
  tam sayıya yuvarlıyor

**Neden iki ayrı fonksiyon:** eğitim sırasında **yumuşak** çıktı gerekiyor — güvenin kademeli
değişmesi, gradient descent'in üzerinde kayabileceği bir yüzey demek. Sınıflandırma sırasında ise
**kesin** cevap gerekiyor, çünkü etiketler zaten 0 veya 1. Aynı model, iki farklı okuma biçimi.

Sigmoid'in kendi davranışı, derivative'i ve `tanh`/ReLU ile karşılaştırması ayrı notta:
[13-aktivasyon-fonksiyonlari](/posts/yz50-13-aktivasyon-fonksiyonlari/).

---

## 4. Loss neden değişmek zorunda *(ders okundu, doğrulandı)*

Bu bölümün en kolay atlanan yeri. "Sigmoid ekledim, MSE'yi tutayım" denebilirmiş gibi görünüyor —
ve Perrotta tam olarak bunu **deniyor**: `loss()` fonksiyonunu olduğu gibi bırakıp sadece
`predict()` yerine `forward()` çağırıyor. Tek satırlık bir değişiklik gibi duruyor.

**Sonra loss yüzeyini çizdiriyor ve iş orada bitiyor.** Yüzey artık "derin kanyonlar ve çukurlar"
ile dolu — yani **yerel minimumlar.** Gradient descent'in tek işi yokuş aşağı inmek; bir çukurun
dibinde aşağısı kalmadığı için orada duruyor ve global minimuma vardığını sanıyor.

![MSE+sigmoid yüzeyi pürüzlü, log loss tek çanak](/yz50/loss-yuzeyi-mse-vs-logloss.png)

İki panel **aynı veri, aynı model** — değişen tek şey loss. Solda yüzey buruşuk ve geniş
platolarla kaplı: o düzlüklerde gradient neredeyse sıfır, gradient descent nereye gideceğini
bilmiyor. Sağda tek bir çanak var ve her noktada aşağı gösteren bir eğim.

*(Bu ölçekte asıl göze çarpan şey platolar; kitaptaki çizimde çukurlar daha belirgin. İkisi aynı
ailenin iki yüzü — biri eğimi sıfırlıyor, diğeri yanlış yere hapsediyor.)*

> **Dersin kendi gerekçesi budur: yüzeyin pürüzlülüğü / yerel minimumlar.** Tek sebep bu.

**Not için önemli bir düzeltme (2026-09-19):** Bu notun ilk hali burada *iki* sebep sayıyordu —
yerel minimumlar **ve** doygunlukta ölen gradient. İkincisi doğru bir olgu ama **bu dersin argümanı
değil**; Perrotta onu burada hiç kullanmıyor. Saturasyon tartışması kursta çok sonra, **Bölüm 19
(*Beyond the Sigmoid*)** geliyor. Senin notlarında zaten var:
[13-aktivasyon-fonksiyonlari](/posts/yz50-13-aktivasyon-fonksiyonlari/). Karıştırma — bu derste öğrenilen
şey yüzey argümanı.

**Çözüm: log loss.**

```
L = -(1/m) * toplam( y*log(tahmin) + (1-y)*log(1-tahmin) )
```

Perrotta'nın vurguladığı sadeleşme: `Y`'deki her etiket 0 ya da 1 olduğu için **her örnek iki
terimden yalnızca birine katkı veriyor.** Etiket 0 ise birinci terim `0` ile çarpılıp yok oluyor;
etiket 1 ise ikinci terim `(1-Y)` ile çarpılıp yok oluyor. Formül korkutucu görünüyor ama kodda
iki satır.

Bunun loss yüzeyi de çizdiriliyor: **kanyon yok, düzlük yok, çukur yok.**

Loss ailelerinin karşılaştırması ve NLL (Negative Log Likelihood)'in istatistiksel anlamı:
[16-loss-nll-cross-entropy](/posts/yz50-16-loss-nll-cross-entropy/). Oradaki "ortalama NLL" ile buradaki
log loss aynı şey — biri ikili, diğeri çok sınıflı hali.

### Gözden kaçmaması gereken bir gerilim

Perrotta sigmoid'i seçerken gerekçesini şöyle koyuyor: gradient descent'in çalışması için sarmalayıcı
fonksiyon **pürüzsüz olmalı, düz bölgesi (gradient'in sıfırlandığı yer) ve boşluğu olmamalı** — ve
sigmoid'in "hiçbir zaman tamamen düzleşmediğini" söylüyor.

Teknik olarak doğru: sigmoid'in derivative'i hiçbir yerde tam olarak sıfır değil. **Pratikte yanıltıcı:**
`z` büyüdükçe derivative o kadar küçülüyor ki öğrenme fiilen duruyor. Bu, YZ50 Hafta 4'te karşına
çıkan **tanh saturation** probleminin ta kendisi — histogramların ±1'e yığılması ve gradient'in
ölmesi. Aynı kursun Bölüm 19'u da buraya geri dönüp sigmoid'i bu yüzden terk ediyor.

Yani: **bu derste sigmoid'in kahraman olduğu argüman, 14 bölüm sonra onun terk edilme sebebi
oluyor.** İkisi çelişmiyor — tek layer'lı bir sınıflandırıcıda sorun değil, derin ağda sorun.

## 5. "Update the Gradient" — bölümün asıl sürprizi *(ders okundu, doğrulandı)*

Yeni bir model, yeni bir loss. Derivative'in de yepyeni ve çirkin olmasını beklersin. Olmuyor.
Perrotta'nın verdiği ifade:

$$\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m} x_i\,(\hat{y}_i - y_i)$$

Ve hemen yanına "şimdiye kadar kullandığımız MSE'nin gradient'i"nı koyuyor:

$$\frac{\partial \text{MSE}}{\partial w} = \frac{2}{m}\sum_{i=1}^{m} x_i\,(\hat{y}_i - y_i)$$

Pratik sonucu: **`gradient()` fonksiyonunu yeniden yazmıyorsun.** Gövdesi aynı kalıyor, sadece
baştaki `2` düşüyor. Değişen tek şey `predict()` yerine `forward()` çağrılması.

> **Perrotta bu ifadeyi türetmiyor** — "here is the partial derivative ... from the math textbooks"
> diyip alıntılıyor. Sadeleşmenin tam ispatı senin kendi notunda:
> [15-aktivasyon-loss-eslesmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/) bölüm 1 — sigmoid'in derivative'indeki
> `ŷ(1-ŷ)` ile log loss'un derivative'indeki `1/(ŷ(1-ŷ))` birebir sadeleşiyor, geriye `(ŷ - y)` kalıyor.

### Dikkat: burada iki farklı MSE var, karıştırma *(2026-09-19'da takılınan nokta)*

"Ama biz MSE'nin gradient'inın farklı olduğunu hesaplamıştık" diye takılmak çok kolay. Ortada
**üç** gradient var:

| | Model | Gradient |
|---|---|---|
| **1. MSE, sigmoid'siz** — lineer regresyon, "şimdiye kadar kullandığımız" | `ŷ = X @ w` | $\frac{2}{m}\sum x_i(\hat{y}_i - y_i)$ |
| **2. MSE, sigmoid'li** — denendi, elendi | `ŷ = sigmoid(X @ w)` | $\frac{2}{m}\sum x_i(\hat{y}_i - y_i)\,\hat{y}_i(1-\hat{y}_i)$ |
| **3. log loss, sigmoid'li** — yeni | `ŷ = sigmoid(X @ w)` | $\frac{1}{m}\sum x_i(\hat{y}_i - y_i)$ |

**Perrotta 3'ü 1 ile karşılaştırıyor** — cümlesindeki anahtar: *"the mean squared error that we
have used so far"*, yani Bölüm 2-4'teki, sigmoid takılmadan önceki MSE. Orada sarmalayıcı yok,
söndürücü çarpan da yok.

**Bölüm 7b'deki hesap ise 3'ü 2 ile karşılaştırıyor**, çünkü "neden MSE değil" sorusunun gerçek
alternatifi 2 numara: sınıflandırma yaparken sigmoid'i çıkarıp 1 numaraya dönemezsin.

İkisi de doğru, farklı soruları cevaplıyor:

- **Perrotta'nın derdi pratik:** "kodu yeniden yazmana gerek yok"
- **7b'nin derdi teorik:** "bu modelde MSE'yi tutsaydın gradient bozulurdu"

### Benzerlik tesadüf değil

Sigmoid, MSE ile eşleştiğinde gradient'i bozuyordu; log loss ile eşleştiğinde **lineer
regresyondaki temiz biçimi geri getiriyor.** Doğru eşleştirme yapıldığında hep aynı şey çıkıyor:

| Problem | Aktivasyon | Loss | `∂L/∂z` |
|---|---|---|---|
| Regresyon | kimlik (yok) | MSE | `ŷ - y` |
| İkili sınıflandırma | sigmoid | log loss | `ŷ - y` |
| Çok sınıflı | softmax | cross-entropy | `ŷ - y` |

Bunun adı **kanonik eşleşme (canonical link)**, çerçevesi GLM:
[15-aktivasyon-loss-eslesmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/).

Üçüncü satır YZ50 Hafta 5'in işi — `dlogits` tam olarak bu. Karpathy'nin "bütün o zincir tek
satıra iniyor" dediği şey, bu tablonun çok sınıflı hali.

### Baştaki 2 nereye gitti

$$\frac{d}{d\hat{y}}\hat{y}^2 = 2\hat{y}$$

MSE'nin karesinden geliyor; log loss'ta kare yok. **Sabit bir çarpan** — yönü değiştirmiyor,
yalnızca adım boyunu ikiye katlıyor, yani learning rate'i yarıya bölmekle aynı şey. Bazı kitaplar
MSE'yi bu yüzden `1/(2m)` ile tanımlıyor, tam olarak o 2'yi kırpmak için.

## 6. Karar sınırı bir hiperdüzlem *(ders henüz okunmadı)*

**Bu bölüm neden var:** Sigmoid bir eğri ekliyor ama karar sınırı hâlâ düz. Bu ayrımı görmeden "nonlineer model" ile "nonlineer karar sınırı" karıştırılıyor.

`sigmoid(z) = 0.5` olduğu yer tam olarak `z = 0` olduğu yer. Yani karar sınırı:

```
w[0] + x1*w[1] + x2*w[2] + ... = 0
```

Bu [02](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/)'deki hiperdüzlemin ta kendisi. Sigmoid eğriyi
ekliyor ama **sınırı eğriltmiyor** — lojistik regresyon hâlâ doğrusal bir sınıflandırıcı.
Eğri sınır istiyorsan layer eklemen gerekiyor, ki bu da kursun 8-11. bölümlerinin (Perceptron →
Designing/Building/Training the Network) konusu.

Bunun sınırı ve "tek layer neyi yapamaz" sorusu: kursun **Bölüm 8, *Where Perceptrons Fail***
dersi (XOR problemi).

---

## 7. Accuracy neden optimize edilmez — ispat

Sınıflandırmada iki ayrı sayı var ve karıştırılıyor:

| | Ne | Nerede kullanılır |
|---|---|---|
| **loss** (log loss) | sürekli, differentiable | **optimize edilir** |
| **accuracy** | doğru bilinen örnek oranı | **raporlanır** |

"Accuracy'yi doğrudan optimize edemezsin" cümlesinin ispatı aşağıda (soru 2026-09-19'da soruldu).

### 7.1 `classify` aslında basamak fonksiyonu

$$\sigma(z) \geq \tfrac{1}{2} \iff \frac{1}{1+e^{-z}} \geq \frac{1}{2} \iff 1+e^{-z} \leq 2 \iff e^{-z} \leq 1 \iff z \geq 0$$

Yani yuvarlama sigmoid'i tamamen atlıyor:

$$c_i = \mathbb{1}[z_i \geq 0] = H(z_i), \qquad H = \text{Heaviside basamak fonksiyonu}$$

**Kritik nokta:** `classify()` çıktısı sigmoid'in *değerine* değil yalnızca **işaretine** bağlı.
Sigmoid'in getirdiği bütün pürüzsüzlük yuvarlama anında çöpe gidiyor.

### 7.2 Heaviside'ın derivative'i — neden sıfır, neden tanımsız

$$H(z) = \begin{cases} 0, & z < 0 \\ 1, & z \geq 0 \end{cases} \qquad H'(z) = \lim_{h \to 0} \frac{H(z+h) - H(z)}{h}$$

**`z = 3` (sıfırdan farklı herhangi bir yer) — derivative sıfır:**

| `h` | `H(3+h)` | pay | oran |
|---|---|---|---|
| `0.1` | `1` | `1-1 = 0` | `0` |
| `-0.5` | `1` | `0` | `0` |
| `-0.001` | `1` | `0` | `0` |

Pay her seferinde sıfır, çünkü 3'ün etrafındaki küçük komşulukta fonksiyon hiç değişmiyor — orada
dümdüz. **`H`, `z = 0` dışında yerel olarak sabittir; sabitin eğimi sıfırdır.** "Çok küçük eğim"
değil, tam olarak sıfır.

**`z = 0` — derivative tanımsız.** `H(0) = 1`. İki taraftan yaklaş:

*Sağdan* (`h > 0`): `H(h) = 1`, oran `(1-1)/h = 0`. Sağ limit **0**.

*Soldan* (`h < 0`): `H(h) = 0`, oran `-1/h`:

| `h` | `-1/h` |
|---|---|
| `-0.1` | `10` |
| `-0.01` | `100` |
| `-0.001` | `1000` |

Sol limit **sonsuz**. İki taraf aynı sayıya gitmiyor → limit yok → derivative yok.

Grafikte: `H` iki yatay çizgi (eğim 0) ve aralarında **dikey bir sıçrama.** Dikey çizginin eğimi
yoktur — "sonsuz dik" demek bir sayı vermek değil.

![Heaviside'ın derivative'i: z=3'te sıfır, z=0'da tanımsız](/yz50/heaviside-turev.gif)

`h` küçüldükçe iki panel farklı davranıyor. **Solda** (`z = 3`) her iki yandaki oran da `0` —
fonksiyon orada dümdüz olduğu için pay hep sıfır. **Sağda** (`z = 0`) sağdaki oran `0`'da
kalırken soldaki `10 → 100 → 1000` diye patlıyor. İki taraf farklı yere gittiği için limit yok.

### 7.3 Sonuç: gradient descent hiç hareket etmiyor

$$a_i = \mathbb{1}[c_i = y_i] = y_i H(z_i) + (1-y_i)\bigl(1-H(z_i)\bigr)$$

$$\frac{\partial a_i}{\partial w} = \underbrace{(2y_i-1)}_{\partial a_i/\partial H} \cdot \underbrace{\frac{dH}{dz}\Big|_{z_i}}_{=\,0} \cdot \underbrace{x_i}_{\partial z_i/\partial w} = \mathbf{0}$$

$$w \leftarrow w - \eta\,\nabla_w A(w) = w - \eta\cdot\mathbf{0} = w$$

Herhangi bir başlangıçtan, herhangi bir learning rate ile, sonsuz adım: **`w` hiç kıpırdamıyor.**
"Yavaş öğrenir" değil, "hiç öğrenmez". Tanımsız olduğu küme (`z = 0` hiperdüzlemi) ölçüsü sıfır —
oraya rastgele düşme probability'nin sıfır, düşsen bile elinde yön bilgisi taşımayan sonsuz bir şey var.

### 7.4 Bu bir kaza değil: sonlu görüntü + süreklilik ⇒ sabit

**"Sonlu görüntü"** = fonksiyonun üretebileceği **farklı çıktı sayısı** sonlu. Bir doğru
(`f(w) = 3w`) karşı örnek değil, sonsuz farklı değer üretiyor. Accuracy ise 100 örnekte yalnızca
101 farklı sayı olabilir: `0.00, 0.01, ..., 1.00`. **`0.315` diye bir accuracy mümkün değil** —
31.5 örneği doğru bilemezsin.

İspat, **ara değer teoremi** ile:

1. Diyelim `A` sürekli ve iki farklı değer alıyor: bir yerde `0.25`, başka yerde `0.50`
2. Ara değer teoremi: sürekli fonksiyon iki değeri alıyorsa **aradaki her değeri de** alır —
   `0.30`, `0.31`, `0.3141592...`
3. Ama `0.25` ile `0.50` arasında **sonsuz** gerçel sayı var, `A`'nın toplam 101 değeri var
4. Çelişki ⇒ `A` iki farklı değer alamaz ⇒ sabit olmalı

`A` sabit **değil** (bazı `w`'ler daha iyi sınıflandırır). Demek ki varsayım yanlıştı: **`A`
sürekli değil.** Sürekli olmayan yerde derivative zaten aranmaz. Bu, 7.2'nin genel hali — `H`'ın
sıçraması tek örnekteki görünüm, buradaki süreksizlik toplamdaki.

![Accuracy basamaklı, log loss pürüzsüz](/yz50/accuracy-merdiven-vs-logloss.gif)

Tek bir parametre soldan sağa oynatılıyor. **Üstte** accuracy bir merdiven: basamakların içinde
tamamen düz (eğim `0`), basamaklar arasında dikey sıçrama (eğim tanımsız). Gradient descent'in
tutunabileceği hiçbir yer yok. **Altta** aynı parametre için log loss — sürekli, tek minimumlu,
her noktada aşağı gösteren bir eğim var.

Bu tek resim bölüm 7'nin tamamını özetliyor.

### 7.5 Bonus: bütün bir yönün gradient'i kesin sıfır

`α` pozitif bir sayı, bütün weight'leri onunla çarp:

```
w = (1, -2, 3)   →   2w = (2, -4, 6)
```

Bir örnekte `z = 4` ise yeni değer `2 * 4 = 8`. **`classify` yalnızca işarete bakıyor:** 4 de
pozitif, 8 de pozitif — tahmin birebir aynı. `z = -4` olsaydı `-8` olurdu, yine aynı.

$$A(\alpha w) = A(w) \quad \text{her } \alpha > 0 \text{ için}$$

`w` yönünde yürüdüğünde (weight'leri hep birlikte büyütüp küçülttüğünde) accuracy kılını
kıpırdatmıyor. Sabit fonksiyonun eğimi sıfır:

$$\nabla A(w)^\top w = 0$$

Gradient sıfır olmasa bile **bu yönde kesinlikle sıfır.** Accuracy `w`'nin büyüklüğünü değil
yalnızca yönünü umursuyor.

**Log loss'ta bu olmuyor** — fark tam burada:

```
sigmoid(4) = 0.982        sigmoid(8) = 0.99966
```

Weight'leri ikiye katlayınca tahmin değişti, loss değişti. Log loss "doğru tarafta mısın"ın
yanında "ne kadar eminsin"i de ölçtüğü için `w`'yi büyütmek gerçek bilgi taşıyor. `round` tam
olarak bu bilgiyi siliyor: `0.982` ve `0.99966`, ikisi de `1`.

### 7.6 Pratik sonuç

**Loss düşerken accuracy bir süre sabit kalabilir.** Model henüz hiçbir örneğin işaretini
değiştirmeden güvenini artırıyordur. Bozuk değil, beklenen davranış.

Log loss bir **vekil (surrogate)** amaç fonksiyonu: asıl umursadığın accuracy ama o
derivativelenemediği için, onunla aynı yöne çeken pürüzsüz bir fonksiyonu minimize ediyorsun.

---

## 7b. MSE neden elendi — accuracy'den tamamen farklı bir sebeple

**Karıştırılması çok kolay, ayrı tutulmalı.** Üç aday vardı, ikisi **farklı sebeplerden** elendi:

| Aday | Neyi kullanıyor | Differentiable mi | Sorunu |
|---|---|---|---|
| accuracy / 0-1 | `classify()` = yuvarlanmış | ❌ **Hayır** | Gradient sıfır (bölüm 7) |
| **MSE** | `forward()` = `sigmoid(z)` | ✅ **Evet** | Gradient **var** ama bozuk |
| log loss | `forward()` = `sigmoid(z)` | ✅ Evet | — |

MSE yuvarlamayı kullanmıyor, sigmoid'in sürekli çıktısını kullanıyor — **pürüzsüz ve
differentiable.** Bölüm 7'deki Heaviside tartışmasının MSE ile hiçbir ilgisi yok.

### 7b.1 Fazladan çarpan

$$\frac{\partial L_{\log}}{\partial z_i} = \frac{1}{m}(\hat{y}_i - y_i)$$

$$\frac{\partial L_{\text{MSE}}}{\partial z_i} = \frac{2}{m}(\hat{y}_i - y_i)\cdot\underbrace{\hat{y}_i(1-\hat{y}_i)}_{\textbf{fazladan}}$$

Log loss'ta neden yok: log loss'un derivative'inde payda olarak `ŷ(1-ŷ)` çıkıyor ve sigmoid'in derivative'iyle
birebir sadeleşiyor. MSE'de sadeleşecek payda yok, çarpan hayatta kalıyor.

### 7b.2 Bu çarpan öğrenme sinyalini ters sıralıyor

Gerçek etiket `y = 0`, model farklı derecelerde yanılıyor:

| `ŷ`     | Durum                | Hata    | Çarpan `ŷ(1-ŷ)` | **MSE gradient'i** | **log loss gradient'i** |
| ------- | -------------------- | ------- | --------------- | ------------------ | ----------------------- |
| `0.5`   | kararsız             | `0.5`   | `0.250`         | `0.125`            | `0.5`                   |
| `0.667` | biraz yanlış         | `0.667` | `0.222`         | **`0.148`**        | `0.667`                 |
| `0.9`   | yanlış               | `0.9`   | `0.090`         | `0.081`            | `0.9`                   |
| `0.99`  | çok emin, çok yanlış | `0.99`  | `0.0099`        | `0.0098`           | `0.99`                  |
| `0.999` | felaket              | `0.999` | `0.000999`      | **`0.000998`**     | `0.999`                 |

Log loss'ta gradient hatayla **büyüyor**. MSE'de `ŷ = 0.667`'de tepe yapıp **çöküyor**: model
tamamen emin ve tamamen yanlışken gradient, *biraz* yanlış olduğu duruma göre **148 kat küçük.**

Tepe noktası (`y=0` için MSE gradient'i `ŷ²(1-ŷ)` ile orantılı):

$$\frac{d}{d\hat{y}}\left[\hat{y}^2 - \hat{y}^3\right] = 2\hat{y} - 3\hat{y}^2 = \hat{y}(2-3\hat{y}) = 0 \ \Rightarrow \ \hat{y} = \tfrac{2}{3}$$

> **MSE + sigmoid, en çok düzeltme gereken yerde en az öğreniyor.**

![MSE gradient'i ŷ=2/3'te tepe yapıp çöküyor](/yz50/mse-gradient-sonmesi.png)

Solda iki gradient yan yana: log loss düz bir doğru gibi hatayla birlikte büyüyor, MSE ise
`ŷ = 2/3`'te tepe yapıp sonra sıfıra çöküyor. Sağda aynı şey log ölçekte, model giderek daha emin
ve daha yanlış hale gelirken — son sütunda MSE'nin çubuğu neredeyse görünmüyor.

### 7b.3 Konvekslik — dersteki çukurların matematiği

$$\nabla^2 L_{\log} = \frac{1}{m}X^\top S X, \qquad S = \mathrm{diag}\bigl(\hat{y}_i(1-\hat{y}_i)\bigr)$$

Herhangi bir `v` için:

$$v^\top \bigl(\nabla^2 L_{\log}\bigr) v = \frac{1}{m}\sum_i \underbrace{\hat{y}_i(1-\hat{y}_i)}_{>\,0}\underbrace{\bigl(x_i^\top v\bigr)^2}_{\geq\,0} \ \geq \ 0$$

Pozitif yarı-tanımlı ⇒ **konveks** ⇒ **yerel minimum yok.** Konveks fonksiyonda bulduğun her
minimum globaldir.

MSE'de bu ispat yürümüyor: sigmoid'in derivative'i ikinci kez derivativelendiğinde işaret değiştirebilen
terimler çıkıyor, Hessian pozitif yarı-tanımlı kalmıyor. Sonuç: dersin çizdirdiği **kanyonlar
ve çukurlar.**

### 7b.4 Peki neden lineer regresyonda MSE sorun değil

Çünkü ölümcül çarpan **sigmoid'den** geliyor, MSE'den değil. Sarmalayıcı yoksa:

$$\hat{y} = z \ \Rightarrow \ \frac{\partial \hat{y}}{\partial z} = 1 \ \Rightarrow \ \nabla_w L_{\text{MSE}} = \frac{2}{m}X^\top(\hat{y}-y)$$

Söndürücü terim yok — log loss'un sınıflandırmada sağladığı temiz davranışın aynısı. Konvekslik
de bedava:

$$v^\top\left(\frac{2}{m}X^\top X\right)v = \frac{2}{m}\lVert Xv\rVert^2 \geq 0$$

Üstelik burada Hessian **sabit**, `w`'ye hiç bağlı değil — her yerde aynı eğrilikte bir çanak.
Kapalı form çözümün var olmasının sebebi bu ([01](/posts/yz50-01-lineer-regresyon/), bölüm 3). Log loss'ta
kapalı form **yok**.

**Ayrıca log loss lineer regresyonda yazılamaz bile:** `log(1 - ŷ)` için `ŷ`'nin 0 ile 1 arasında
olması gerekiyor; regresyonda `ŷ = 17` çıkabilir, `log(-16)` tanımsız. Ve `y = 23 pizza` etiketi
formüle girmiyor. Reddetmedik — **kullanılabilir değildi.**

### 7b.5 İki soru, iki ayrı cevap

- **"Neden accuracy değil"** → derivative'i yok, gradient descent hiç hareket etmiyor
- **"Neden MSE değil"** → derivative'i var ama `ŷ(1-ŷ)` çarpanı hem öğrenme sinyalini en kritik anda
  söndürüyor hem yüzeyi konveks olmaktan çıkarıyor

Peki loss'u kim seçiyor, neden bu ikisi? →
[16-loss-nll-cross-entropy](/posts/yz50-16-loss-nll-cross-entropy/), "Loss nereden geliyor" bölümü.

## 7c. Accuracy ölçerken de yanıltıcı — MNIST deneyi

*(2026-09-19, Perrotta Bölüm 6'nın "hangi rakam zor" tablosu üzerine yapılan ek analiz.
Kod: [`more_content/F_digit_error_analysis.py`](https://github.com/CoYildiz/yz50/blob/main/more_content/F_digit_error_analysis.py))*

Bölüm 7 accuracy'nin **optimize edilemeyeceğini** gösteriyordu. Bu bölüm daha sinsi bir şeyi:
accuracy **raporlanırken de** yanıltıyor.

### Sınıf dengesizliği başlangıç çizgisini yukarı itiyor

Her sınıflandırıcı "bu bir 8 mi" diye soruyor ve test setinin yalnızca ~%10'u 8. Yani **hiçbir şey
öğrenmeyip her örneğe "hayır" diyen** model bile ~%90 alıyor. Başlangıç çizgisi %50 değil, %90.

| Rakam | Adet | "Hep hayır" der | Model | **Kazanç** | FN | FP | **Recall** | Precision |
|---|---|---|---|---|---|---|---|---|
| 0 | 980 | 90.20% | 98.99% | +8.79 | 49 | 52 | 95.0% | 94.7% |
| 1 | 1135 | 88.65% | 99.03% | **+10.38** | 64 | 33 | 94.4% | 97.0% |
| 2 | 1032 | 89.68% | 97.37% | +7.69 | 212 | 51 | 79.5% | 94.1% |
| 3 | 1010 | 89.90% | 96.98% | +7.08 | 220 | 82 | 78.2% | 90.6% |
| 4 | 982 | 90.18% | 97.59% | +7.41 | 177 | 64 | 82.0% | 92.6% |
| 5 | 892 | 91.08% | 96.37% | +5.29 | 295 | 68 | 66.9% | 89.8% |
| 6 | 958 | 90.42% | 98.07% | +7.65 | 114 | 79 | 88.1% | 91.4% |
| 7 | 1028 | 89.72% | 98.14% | +8.42 | 138 | 48 | 86.6% | 94.9% |
| **8** | 974 | 90.26% | 93.85% | **+3.59** | **419** | 196 | **57.0%** | 73.9% |
| 9 | 1009 | 89.91% | 95.57% | +5.66 | 300 | 143 | 70.3% | 83.2% |

Ham accuracy'de "93.85 vs 99.03" küçük bir fark gibi duruyor. Kazanca bakınca 8 sınıflandırıcısı
hiçbir şey yapmamaya göre **3.6 puan**, 1 sınıflandırıcısı **10.4 puan** kazanmış — yani 8'in
öğrendiği şey 1'in öğrendiğinin üçte biri.

5'e de dikkat: ham accuracy'de ortalarda (96.37) ama kazançta sondan ikinci, çünkü test setinde en
az 5 var (892) ve başlangıç çizgisi en yüksek onda.

### Önce terimler: confusion matrix

**Bu alt bölüm neden var:** Aşağıda FN, FP, recall, precision geçiyor ve bunlar birbirine
karıştırılmaya çok uygun. Dördünü tek tabloda görmek karışıklığı bitiriyor.

Her tahmin dört kutudan birine düşüyor:

| | **Model "evet" dedi** | **Model "hayır" dedi** |
|---|---|---|
| **Gerçekte evet** | **TP** (true positive) — doğru yakaladı | **FN** (false negative) — **kaçırdı** |
| **Gerçekte hayır** | **FP** (false positive) — **yanlış alarm** | **TN** (true negative) — doğru eledi |

Okuma kuralı: ikinci kelime **modelin ne dediği** (positive = "evet"), birinci kelime **doğru mu
olduğu**. "False negative" = model "hayır" dedi ve yanıldı.

Buradan çıkan ölçüler:

| Ölçü | Formül | Hangi soruyu soruyor |
|---|---|---|
| **accuracy** | $\dfrac{TP+TN}{\text{hepsi}}$ | Tüm tahminlerin kaçı doğruydu |
| **precision** | $\dfrac{TP}{TP+FP}$ | **"Evet" dediklerimin** kaçı gerçekten evetti |
| **recall** (sensitivity) | $\dfrac{TP}{TP+FN}$ | **Gerçek evetlerin** kaçını yakaladım |
| **specificity** | $\dfrac{TN}{TN+FP}$ | Gerçek hayırların kaçını doğru eledim |
| **F1** | $2\cdot\dfrac{P \cdot R}{P+R}$ | Precision ve recall'un harmonik ortalaması — ikisini tek sayıda birleştirir |
| **balanced accuracy** | $\dfrac{\text{recall} + \text{specificity}}{2}$ | İki sınıfa **eşit ağırlık** veren accuracy |

**Precision ile recall birbirinin bedeli.** Eşiği düşürürsen daha çok "evet" dersin: recall
yükselir, precision düşer. Hangisinin öncelikli olduğu probleme bağlı:

- **Duman dedektörü** → recall öncelikli. Yangını kaçırmak felaket, boşa alarm katlanılabilir
- **Ceza yargısı** → precision öncelikli. Masum birini mahkûm etmek, suçlunun serbest kalmasından ağır

**F1 neden harmonik ortalama:** aritmetik ortalama olsaydı precision %100 / recall %0 olan bir
model %50 alırdı — oysa o model hiçbir işe yaramıyor. Harmonik ortalama küçük olanı cezalandırıyor:
aynı modelin F1'i **0**.

### Kendi sayılarınla aynı tablo

| Rakam | TP | FN | FP | TN | accuracy | precision | recall | **F1** | **balanced acc** |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 1071 | 64 | 33 | 8832 | 99.03% | 97.0% | 94.4% | **95.7%** | **97.0%** |
| **5** | 597 | 295 | 68 | 9040 | 96.37% | 89.8% | 66.9% | **76.7%** | **83.1%** |
| **8** | 555 | 419 | 196 | 8830 | 93.85% | 73.9% | 57.0% | **64.3%** | **77.4%** |

**Son iki sütun asıl hikâyeyi anlatıyor.** Accuracy'de 8 ile 1 arasında 5 puan fark var
(93.85 vs 99.03) — küçük görünüyor. F1'de **31 puan** (64.3 vs 95.7), balanced accuracy'de
**20 puan** (77.4 vs 97.0).

Sebep: accuracy'nin payında `TN` var ve 8 sınıflandırıcısının `TN`'i 8830 — yani doğru bildiği
"8 değil" örnekleri payı şişiriyor. Precision, recall ve F1'in formüllerinde `TN` **hiç geçmiyor**;
tam da bu yüzden dengesiz veride bilgilendiriciler.

> **Kural:** dengesiz veride accuracy raporlamak yanıltıcı. En az **recall + precision** (ya da
> ikisini birleştiren **F1**) ver, ve "hiçbir şey yapmayan model ne alırdı" çizgisini yaz.
>
> *(Kaynak: [Precision and recall — Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall).
> Yukarıdaki sayılar [`more_content/F_digit_error_analysis.py`](https://github.com/CoYildiz/yz50/blob/main/more_content/F_digit_error_analysis.py) çıktısından hesaplandı.)*

### Asıl gizlenen sayı: recall

8 için accuracy `%93.85`, **recall `%57`.** Model gerçek 8'lerin ancak yarısından biraz fazlasını
yakalıyor. Accuracy bunu tamamen gizliyor, çünkü doğru bildiği 9026 tane "8 değil" örneği payda
şişiriyor.

| Metrik | Neyi sorar |
|---|---|
| **accuracy** | tüm örneklerin kaçını doğru bildim |
| **recall** | gerçek 8'lerin kaçını **yakaladım** |
| **precision** | "8" dediklerimin kaçı **gerçekten** 8'di |

### Ders "hangi hata" sorusunu bırakmıştı — cevap: kaçırma

```
En zor rakam: 8
  kaçırdığı (FN)   : 419  / 974 gerçek 8      ← baskın hata
  yanlış evet (FP) : 196
```

Model 8'leri başka rakamlarla **karıştırmıyor**, 8'leri **bulamıyor.** FP dağılımı yine de
öğretici:

```
gerçek 5:  73 kez 8 sanıldı
gerçek 2:  46 kez 8 sanıldı
gerçek 1:  30 kez 8 sanıldı
```

5 ↔ 8 çift yönlü karışıyor — 5'in kendi recall'u da `%66.9` ile sondan ikinci. İkisi de kapalı
ilmek + kıvrım içeriyor, piksel uzayında iç içe geçiyorlar.

### Neden 8 zor, 1 kolay — doğrusal modelin gerçek sınırı

Model tek layer'lı ve ham piksellerle çalışıyor, yani öğrendiği şey pratikte bir **şablon**:
`w`'nin ilk elemanını atıp kalan 784'ü `28×28`'e geri sararsan o rakamın weight haritasını resim
olarak görebilirsin.

- **1** ince, dikey, hep aynı bölgeden geçen bir çizgi — az sayıda pikselde yoğunlaşıyor, başka
  hiçbir rakam o bölgeyi öyle doldurmuyor → şablon çok ayırt edici
- **8** neredeyse bütün kareyi dolduruyor ve 5, 2, 3, 0 ile **piksel düzeyinde** büyük örtüşme var.
  Doğrusal bir sınır bunları ayıramıyor

Dersin "insan sezgisi 7 ve 4 zor olur der, yanlış" gözleminin sebebi bu: doğrusal model için
önemli olan **el yazısı çeşitliliği** değil, **diğer sınıflarla piksel örtüşmesi.** İnsan sezgisi
birinciye bakıyor, model ikinciye takılıyor.

Bu, [05-ag-analizi](/posts/yz50-05-ag-analizi-ve-sinirlari/)'deki "weight'ler beklediğin özellik dedektörü
gibi davranmıyor" gözleminin tek layer'lı hali. Aşmanın yolu layer eklemek (kursta Bölüm 8-11)
ve sonunda komşuluğu modele öğretmek (CNN, Bölüm 20).

### Alınacak genel ders

> **Dengesiz veride accuracy neredeyse hiçbir şey söylemez.** Her zaman iki soruyu sor:
> *hiçbir şey yapmayan model ne alırdı*, ve *recall kaç*.

Bu, fraud detection gibi problemlerde (roadmap Faz 3) doğrudan karşına çıkacak: dolandırıcılık
oranı %0.1 ise "hep temiz" diyen model %99.9 accuracy alır ve tamamen işe yaramazdır.

---

## 8. Okurken cevapla

Bölümü bitirdiğinde bu beşini kendi cümlelerinle cevaplayabiliyor olmalısın. Cevaplayamadığını

- [ ] Sınırın **doğru** tarafında, çok uzakta duran bir örnek lineer regresyonu neden bozuyor?
      Sigmoid bunu tam olarak nasıl çözüyor?
- [ ] MSE'yi sigmoid'le birlikte kullanınca loss yüzeyine ne oluyor, ve bu neden gradient
      descent'i bozuyor? (Doğru cevapta "yerel minimum" kelimesi geçmeli)
- [ ] `forward()` ve `classify()` neden iki ayrı fonksiyon? Hangisi eğitimde, hangisi
      sınıflandırmada kullanılıyor ve neden?
- [ ] `gradient` fonksiyonunun gövdesi neden değişmedi? Hangi iki terim sadeleşti?
- [ ] Karar sınırı neden `z = 0`, ve neden hâlâ düz?
- [ ] Loss düşerken accuracy neden bir süre sabit kalabilir?

**Okurken doğrulanacaklar** — *Invasion of the Sigmoids* okundu, ilk ikisi kapandı:

- [x] **Perrotta log loss'a bu derste mi geçiyor, yoksa önce MSE'yi sigmoid'le mi deniyor?**
      Önce MSE'yi deniyor, sonra loss yüzeyini çizdirip bırakıyor. Gerekçe **yerel minimumlar**;
      saturasyon argümanını burada hiç kullanmıyor
- [x] **Log loss'taki katsayı ne?** `-(1/m)`, yani ortalama. (`2/m` katsayısı MSE'nin derivative'ine
      aitti, karıştırma)
- [ ] *Linear Regression Limitation*: lineer regresyonun kırılma sebebi olarak kaç tane ve hangi
      argüman veriliyor? Bu notun 2. bölümü üçünü sayıyor — hepsi kitapta var mı?
- [x] **Update the Gradient: gradient'in biçimi aynı mı kalıyor?** Evet — `(1/m) Σ x(ŷ-y)`.
      Perrotta **türetmiyor**, "math textbooks"tan alıntılıyor. Ve karşılaştırdığı MSE,
      *sigmoid'li* değil *lineer regresyondaki* MSE — bkz. bölüm 5'teki üç gradient tablosu
- [ ] *Classification in Action*: karar sınırının `z = 0` olduğu açıkça söyleniyor mu?
- [ ] "Weighty Decisions" playground'ı hangi parametreyi oynatıyor, ne göstermeye çalışıyor?

---

## İlgili notlar

- [01-lineer-regresyon](/posts/yz50-01-lineer-regresyon/) — bu bölümün kırdığı model; MSE'nin Gauss gürültü
  altında MLE olması, sınıflandırmada bu gerekçenin neden düştüğü
- [02-hiperuzay](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/) — bias hilesi, matrix gradient'i ve
  karar sınırının geometrisi; gradient biçiminin neden değişmediğinin zemini
- [13-aktivasyon-fonksiyonlari](/posts/yz50-13-aktivasyon-fonksiyonlari/) — sigmoid'in derivative'i, saturasyon,
  tanh/ReLU karşılaştırması
- [15-aktivasyon-loss-eslesmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/) — **sigmoid + log loss
  sadeleşmesinin tam ispatı**; bu bölümün "Update the Gradient" dersinin arkasındaki matematik
- [16-loss-nll-cross-entropy](/posts/yz50-16-loss-nll-cross-entropy/) — loss aileleri, NLL'in istatistiksel
  karşılığı, çok sınıflı hale genişleme

## Kursta sırada ne var

Bu bölümden sonra YZ50 ile kesişen ilk duraklar: **Bölüm 7** (One Hot Encoding), **Bölüm 9**
(Introduction to Softmax), **Bölüm 10** (Cross Entropy) — üçü birlikte Hafta 3'ün loss tarafı;
ve **Bölüm 11** (From the Chain Rule to Backpropagation), Hafta 5'in en yakın yazılı karşılığı.

Görseller [img/make_classification_figures.py](/yz50/make_classification_figures.py) ile üretiliyor.
