---
title: "16 · Loss, NLL ve Cross-Entropy"
published: 2026-09-29
description: "Loss fonksiyonunun olasılıktan türetilmesi: likelihood, negative log-likelihood ve cross-entropy'nin aynı şeyin üç adı oluşu."
tags:
  - YZ50
  - Loss
  - Cross-Entropy
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

Loss'un ne olduğu, iki farklı loss ailesinin ne sorduğu, ve NLL ortalamasının istatistiksel karşılığı.

---

## Loss tek bir soru sorar

**Bu bölüm neden var:** "Loss" kelimesi tek bir şeymiş gibi kullanılıyor ama iki farklı soru soran iki ayrı aile var. Hangisinin hangi problemde kullanıldığı bu ayrıma bağlı.

**"Ne kadar yanlıştım?"** Cevabı tek bir sayı. Bu sorunun iki farklı soruş biçimi var.

### 1. Fark alma — kare hata (MSE)

```
model dedi : 0.7
gerçek     : 1.0
fark       : -0.3
karesi     : 0.09    ← loss
```

Buradaki `1.0` **etiket** — veri setinde yazan gerçek cevap. Beklenen değer değil, rastgele değişken değil, sabit bir sayı.

Kare almanın iki sebebi: işareti yok eder (fazla ve eksik tahmin eşit cezalanır), ve derivative'i süreklidir (mutlak değerin köşesi yoktur).

### 2. Olasılığa bakma — negative log likelihood (NLL)

Burada **fark alınmıyor**:

```
model dedi : a=%50, b=%30, c=%20
gerçek     : "b"
loss = -log(0.30) = 1.20
```

![NLL bigram bigram nasıl birikiyor](/yz50/nll-birikimi.gif)

Aynı durumun farklı modelleri:

| Model "b"ye ne verdi | Loss |
|---|---|
| 0.90 | $-\log 0.90 = 0.11$ |
| 0.30 | $-\log 0.30 = 1.20$ |
| 0.01 | $-\log 0.01 = 4.61$ |

Tek soru: **gerçekten olan şeye ne probability verdin?** Yüksek verdiysen ceza küçük, düşük verdiysen büyük. Sıfır verdiysen ceza sonsuz — smoothing'in gerekçesi bu.

---

## Beklenen değer nerede

**Bu bölüm neden var:** NLL'i "logaritma al, ortalama al" diye uygulamak mümkün ama o ortalamanın neyin ortalaması olduğu belirsiz kalıyor. Bu bölüm onu adlandırıyor — ve adlandırınca perplexity'nin nereden geldiği de açılıyor.

Loss'un formülünün **içinde** değil, loss'ların **üstünde**. Her örnek için bir loss hesaplanır, sonra ortalaması alınır:

```
örnek 1 → 0.09
örnek 2 → 1.20
örnek 3 → 0.11
ortalama = 0.47    ← modelin toplam loss'u
```

Bu ortalama, istatistikteki **ampirik beklenen değer**. Kodda karşılığı `log_likelihood / n`.

### Ampirik ortalama vs gerçek beklenen değer

Hesapladığın şey:

$$\hat{L} = \frac{1}{n}\sum_{k=1}^{n} \big(-\log q(y_k \mid x_k)\big)$$

Tahmin etmeye çalıştığı şey:

$$L = \mathbb{E}_{(x,y)\sim p}\big[-\log q(y \mid x)\big]$$

Burada $p$ gerçek dağılım (bilinmiyor, sadece örnekleri görülüyor), $q$ model. İlişki standart: ampirik ortalama, beklenen değerin **yansız tahmin edicisi**; örnek sayısı arttıkça büyük sayılar yasasıyla gerçek değere yakınsıyor.

---

## Bu beklenen değerin adı: cross-entropy

**Bu bölüm neden var:** Cross-entropy, NLL ve log loss aynı şeyin üç adı — literatürde üçü de geçiyor ve karıştırılıyor. Aradaki ilişkiyi bir kez kurmak gerekiyor.

$$\mathbb{E}_{y\sim p}\big[-\log q(y)\big] = H(p, q)$$

Yani NLL ortalaması, gerçek dağılım ile modelin dağılımı arasındaki **cross-entropy**.

Açılımı:

$$H(p,q) = H(p) + D_{KL}(p \,\|\, q)$$

- $H(p)$ — verinin kendi entropisi. Sabit, modelin ne yaptığından bağımsız.
- $D_{KL}(p\|q)$ — modelin gerçek dağılımdan sapması. Her zaman $\ge 0$, ancak $p=q$ ise sıfır.

**Sonuç:** loss'u küçültmek = KL ıraksamasını küçültmek. Ulaşılabilecek en düşük değer $H(p)$ — mükemmel bir model bile sıfır loss alamaz, çünkü dilin kendi belirsizliği var.

### Neden adı "likelihood"

NLL'i minimize etmek, likelihood'u maksimize etmekle aynı şey — istatistikteki **maximum likelihood estimation**. Log almak çarpımı toplama çeviriyor (sayısal olarak da gerekli), eksi işareti maksimizasyonu minimizasyona.

### One-hot etiket neden tek terime çöküyor

Cross-entropy tanımı bütün sınıflar üzerinden bir toplam:

$$H(p,q) = -\sum_i p_i \log q_i$$

Gerçek etiket one-hot olduğunda ($p$ = doğru sınıfta 1, diğerlerinde 0) toplamdaki bütün terimler sıfırla çarpılıp düşüyor, geriye tek terim kalıyor: doğru sınıfın log probability'si. Kodda `log_prob = torch.log(prob)` satırı bu.

---

## Sayının okunabilir hali: perplexity

**Bu bölüm neden var:** Loss `2.45` bir insana hiçbir şey söylemiyor. Perplexity aynı bilgiyi "model kaç seçenek arasında kararsız" diye okunabilir hale getiriyor — dil modeli makalelerinin bu sayıyı raporlamasının sebebi bu.

$$\text{perplexity} = e^{\text{NLL}}$$

| Model | NLL | perplexity |
|---|---|---|
| Bigram (senin) | 2.4540 | 11.63 |
| Hiçbir şey bilmeyen (düzgün dağılım) | 3.2958 | 27.00 |

Okunuşu: model her adımda ortalama **11.63 harf arasında kararsız**. Hiçbir şey bilmeyen bir model 27 harf arasında kararsız olurdu. Bigram tablosu 27'yi 11.6'ya indirmiş.

Hafta 4'ün MLP (Multi-Layer Perceptron)'si bu sayıyı daha da aşağı çekmeli — karşılaştırma ölçütü bu.

---

## Sayısal kararlılık: log loss'u `nan` yapan şey doğru tahmin

*(2026-09-19, Perrotta Bölüm 6'daki `clip` uyarısı üzerine)*

Log loss içinde `log(ŷ)` ve `log(1-ŷ)` var. Matematikte sigmoid asla tam `0` veya `1` olmadığı için
sorun yok. **Float64'te oluyor.**

### Sigmoid gerçekten 1.0 döndürüyor

```
sigmoid(36.7) = 0.9999999999999998    → 1.0 değil
sigmoid(36.8) = 1.0                   → tam olarak 1.0
```

`z` yaklaşık `36.8`'i geçince `exp(-z)` o kadar küçülüyor ki `1 + exp(-z)` float64'te `1`'e
yuvarlanıyor. Diğer uçta `z < -745` olunca `exp(-z)` taşıyor ve sigmoid `0.0` oluyor.

### `nan`'ı üreten şey, yanlış değil **doğru** tahmin

`y = 1`, `ŷ = 1.0` — model tamamen haklı ve tamamen emin:

```
first_term  = 1 * log(1.0)     = 0.0                      ✓
second_term = (1-1) * log(1-1) = 0 * (-inf)  = nan        ✗
```

Matematikte `(1-y)` çarpanı sıfır olduğu için o terimin yok sayılması gerekirdi. Ama IEEE kayan
noktada **`0 × sonsuz = nan`**, ve tek bir `nan` ortalamaya girince bütün loss `nan` oluyor.

![Sigmoid float64'te 1.0'a yapışıyor ve nan doğuyor](/yz50/logloss-nan-dogru-tahmin.png)

Solda `1 − sigmoid(z)`'nin `z ≈ 36.7`'den sonra **tam olarak sıfır** olması. Sağda sonucu: etiket
`y = 1` ve model `ŷ = 1.0` derken — yani **tamamen haklıyken** — ikinci terim `0 × (−∞)` oluyor.
Diğer dört sütunda aynı terim sorunsuz `0`.

> Uyarı "model kötü" demiyor — tam tersine, **model o kadar iyi ki aritmetik kırılıyor.**

### Çözüm: uçlardan bir tık içeri it

```
y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
```

```
clip öncesi:  toplam = nan
clip sonrası: toplam = -1e-15      (pratikte 0 — doğru cevap)
```

`1e-15` float64'ün hassasiyetinden geliyor: ~15-16 anlamlı basamak, yani `1`'den ayırt
edilebilecek en yakın mesafe bu civarda.

**Bedeli:** log loss teoride "emin ve yanlışsan ceza sonsuz" diyordu; clip bunu tavanlıyor.

```
-log(1e-15) = 34.54     ← alınabilecek en büyük ceza
```

Sonsuz ceza zaten hesaplanabilir bir şey değildi.

### Neden eğitim yine de bozulmuyor

Gradient'in içinde logaritma **yok**:

```
gradient = X.T @ (y_hat - Y) / m
```

`ŷ = 1.0` olsa bile bu ifade sağlıklı. Yani `w` güncellenmeye devam ediyor, yalnızca **ekrana
yazdırılan sayı** `nan` oluyor. Öğrenirken kozmetik bir sorun.

Üretimde kozmetik değil: loss'a bakıp early stopping yapıyorsan, en iyi modeli loss'a göre
seçiyorsan veya loss'u logluyorsan, `nan` o mekanizmaların hepsini bozar.

### Aynı problemin YZ50'deki hali — ama farklı çözüm

`F.cross_entropy`'nin logits'ten maksimumu çıkarmasının sebebi bu ailenin başka bir üyesi:
`exp(logits)` büyük değerlerde **taşıyor**. Hafta 5'te bu çıkarma `logit_maxes` diye ayrı bir ara
değişken olarak karşına çıkıyor ve gradient'inı elle yazıyorsun (sonucu ~0 çıkar).

İki yaklaşım aynı şey değil:

| | Ne yapıyor | Ne zaman |
|---|---|---|
| **`clip`** | taşma **olduktan sonra** sonucu kırpıyor | kaba, loss'u tavanlıyor |
| **maksimumu çıkarma** (log-sum-exp) | taşmayı **oluşmadan önce** engelliyor | matematiksel olarak birebir aynı sonuç |

İkincisi üstün: hiçbir bilgi kaybetmiyor, çünkü softmax'tan maksimumu çıkarmak sonucu
matematiksel olarak değiştirmiyor. `clip` ise gerçekten bir şeyi kesiyor.

---

## Loss nereden geliyor — maximum likelihood

*(2026-09-19 eklendi. "Neden MSE değil de log loss" sorusu buraya kadar indi: loss'u kim seçiyor?)*

MSE neden kare alıyor, NLL neden logaritma? İkisi de uydurma değil — **tek bir sorudan** türüyor.

### Soru ters duruyor, önce onu düzelt

> "Elimdeki veriyi görme probability'sim ne kadar?"

Veri zaten elinde, olmuş bitmiş; probability'si ne demek? **Numara şu: veriyi puanlamıyoruz, modeli
puanlıyoruz. Veri sadece puan cetveli.**

### ML'siz bir örnek — iki kutu para

- **A kutusu:** bu paralar %90 tura geliyor
- **B kutusu:** bu paralar %10 tura geliyor

Biri sana hangi kutudan olduğunu söylemeden bir para veriyor. 10 atış: **9 tura, 1 yazı.**
Hangi kutu? A, açık. Ama **neden**? Sezgiyi sayıya çevir:

$$P(\text{9 tura 1 yazı} \mid A) = 0.9^9 \times 0.1 \approx 0.0387$$

$$P(\text{9 tura 1 yazı} \mid B) = 0.1^9 \times 0.9 \approx 0.0000000009$$

A, gördüğün şeyi yaklaşık **43 milyon kat** daha olası kılıyor. O yüzden A diyorsun.

![İki kutu: hangisi gördüğümü açıklıyor](/yz50/olabilirlik-iki-kutu.gif)

Atışlar birikirken iki çubuk açılıyor. **Veri değişmiyor** — aynı 9 tura 1 yazı. Değişen tek şey
hangi hipotezin bunu ne kadar beklenir kıldığı. Sağdaki sayı her atışta büyüyor, çünkü her yeni
tura A lehine bir kanıt daha ekliyor.

Ne yaptığına dikkat:

- **Veriyi değiştirmedin.** 9 tura 1 yazı sabit
- Değişen şey **hipotezdi**: A mı B mi
- Her hipotez için "bu doğru olsaydı gördüğüm şey ne kadar beklenir olurdu" diye sordun
- **Gördüğünü en az şaşırtıcı kılan** hipotezi seçtin

"Probability" kelimesi burada veriyi değil **hipotezi** derecelendiriyor. Veri terazinin sabit tarafı.

> Terim ayrımı: hipotez sabitken veri değişiyorsa **probability**; veri sabitken hipotez değişiyorsa
> **olabilirlik (likelihood)**. Aynı sayı, ters okunuşu.

### Aynısı, model için

Tek fark: iki kutu değil **sonsuz** kutu var — her `w` değeri bir kutu. Model bir tahmin makinesi
değil, bir **üretici**: `ŷ = 0.9` demek *"bu örneği tekrar tekrar üretsem %90'ında etiket 1
çıkardı"*.

3 örnek, gerçek etiketler `1, 1, 0`:

| Aday | Tahminler | Elindeki etiketlere verdiği probability |
|---|---|---|
| **A** | `0.9, 0.8, 0.1` | `0.9 × 0.8 × 0.9 = 0.648` |
| **B** | `0.3, 0.4, 0.7` | `0.3 × 0.4 × 0.3 = 0.036` |

A, gözlenen etiketleri 18 kat daha olası kılıyor → **A daha iyi model.** Para kutusuyla birebir
aynı akıl yürütme.

**Eğitim = bu skoru en büyük yapan `w`'yi aramak.**

### İkili durum: log loss buradan çıkıyor

Tek örneğin probability'si bir if-else, ama tek formülde yazılabiliyor:

$$p = \hat{y}^{\,y}(1-\hat{y})^{\,1-y}$$

- `y = 1` → üsler 1 ve 0 → `0.9 × 1 = 0.9` ✓
- `y = 0` → üsler 0 ve 1 → `1 × 0.1 = 0.1` ✓

(Aynı anahtar numarası log loss'un iki teriminde de var — her örnek yalnızca birine katkı veriyor.)

Bütün veri için probability'leri **çarp**. İki pratik sorun: 1000 tane `0.9` çarpılınca sayı sıfıra
çöküyor, ve çarpımın derivative'i berbat. Çözüm logaritma — çarpımı toplama çeviriyor:

$$\log p_{\text{toplam}} = \sum_i \Bigl[y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Bigr]$$

Bunu **maksimize** etmek istiyoruz; optimizasyon araçları minimize eder, o yüzden eksiyle çarp:

$$L = -\frac{1}{m}\sum_i \Bigl[y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Bigr]$$

**Bu log loss.** Kimse tasarlamadı — "veriyi en olası kılan parametreleri bul" cümlesinin
matematiksel hali. Adı da bunu söylüyor: **negative log likelihood**, bu notun 2. bölümündeki NLL.

Yani NLL "modelin ne kadar yanlış olduğu" değil, tam olarak şu: **model, gerçekte olmuş olan şeye
ne kadar düşük probability verdi.** Olmuş bir şeye düşük probability veren model kötü modeldir.

### Sürekli durum: MSE buradan çıkıyor

Burada bir pürüz var: "tam olarak 45 pizza gelme probability'si" sıfırdır (45.0000… tam tutmalıydı).
O yüzden probability yerine **çan eğrisinin o noktadaki yüksekliğine** bakılıyor.

Gerçek değer **45**, iki aday:

| Aday | Tahmin | 45 nereye düşüyor | Yükseklik (yayılım 5) |
|---|---|---|---|
| **A** | `42` | tepeye çok yakın | `exp(-0.18) ≈ 0.84` |
| **B** | `10` | kuyrukta | `exp(-24.5) ≈ 0.00000000002` |

Aynı mantık: **gördüğün değeri kuyrukta bırakan model kötü modeldir.**

![Çan eğrisinin yüksekliği ve tahminin kayması](/yz50/can-egrisi-olabilirlik.gif)

Kırmızı çizgi sabit (gerçek değer 45); hareket eden şey **modelin tahmini**, yani çanın tepesi.
Turuncu dikme 45'in o çan altındaki yüksekliği — model uzaktayken neredeyse sıfır, tahmin 45'e
yaklaştıkça tepeye çıkıyor. Sağdaki eğri bu yüksekliğin tahminle nasıl değiştiğini izliyor:
**logaritmasını al, `exp` gidiyor ve geriye `−(y−ŷ)²` kalıyor.** MSE'nin karesi buradan.

$$p \ \propto \ \exp\left(-\frac{(y-\hat{y})^2}{2\sigma^2}\right) \quad\Longrightarrow\quad \log p = -\frac{(y-\hat{y})^2}{2\sigma^2} + \text{sabit}$$

Maksimize etmek, baştaki eksi yüzünden, kare hataların toplamını minimize etmek demek — **MSE.**
Kare almanın sebebi "negatifler birbirini götürmesin" değil, **çan eğrisinin üssünde kare olması.**

### Sonuç: loss serbest bir tercih değil

| Çıktı ne | Varsayım | Çıkan loss |
|---|---|---|
| Sayı (pizza, fiyat) | tahmin + çan eğrisi gürültü | **MSE** |
| İkili (evet/hayır) | probabilitylı yazı-tura | **log loss / NLL** |
| Çok sınıflı (hangi harf) | kategorik dağılım | **cross-entropy** |

MSE "kötü loss", log loss "iyi loss" değil. **İkisi de aynı prensibin, farklı çıktı tipleri için
verilmiş doğru cevabı.** Hata, birini diğerinin alanında kullanmak.

**Sigmoid + MSE'nin tutarsızlığı da buradan:** sigmoid "çıktım bir probability, etiket ikili" diyor,
MSE "çıktının etrafında sürekli bir çan eğrisi gürültüsü var" diyor. Model ve loss, veri hakkında
**iki farklı hikâye** anlatıyor. [03](/posts/yz50-03-siniflandirmaya-gecis/) bölüm 7b'de hesaplanan
bozukluklar — sönen gradient, yerel minimumlu yüzey — bu tutarsızlığın **belirtileri**, sebebi değil.

### Üç layer'lı seçim — ne kimin kararı

| | Ne seçiliyor | Kim seçiyor |
|---|---|---|
| **Hipotez ailesi** | "doğrusal toplam + sigmoid" biçimi | **sen**, modeli kurarken |
| **Loss** | puanlama cetveli | **çıktı tipi** dayatıyor (yukarıdaki tablo) |
| **`w`** | o ailedeki belirli bir üye | **veri**, gradient descent yoluyla |

`w = (1, -2, 3)` bir hipotez, `w = (0.5, 4, -1)` başkası — ikisi de aynı ailenin üyesi. Para kutusu
örneğindeki "A kutusu / B kutusu" bunlara denk geliyor; orada iki taneydi, burada sonsuz.

**Loss hipotezin parçası değil**, hipotezin puanı.

### Neden tek tek deneyip en iyisini seçmiyoruz

Para kutusu örneğinde iki adayı hesaplayıp karşılaştırdın. Burada yapamazsın: `w` sürekli ve çok
boyutlu, aday sayısı sonsuz. Tek tek deneyemediğin için **eğime bakıp yokuş aşağı yürüyorsun.**

Bu zaten ayrı bir notta: [07 · Neden Gradient Descent?](/posts/yz50-07-neden-gradient-descent/) — boyutun laneti
ve rastgele aramanın neden çalışmadığı. O not "neden arama değil"i, bu bölüm "aradığımız şey
neydi"yi cevaplıyor. Aynı resmin iki yarısı.

---

## Hangi loss neye yakınsar

**Bu bölüm neden var:** Loss seçmek yalnızca "hata nasıl ölçülür" değil, **modelin hedef dağılımın hangi istatistiğini öğreneceği** kararı. MSE ortalamayı, MAE medyanı öğretiyor — bu bir yan etki değil, seçimin doğrudan sonucu.

Loss seçimi, modelin hedef dağılımın **hangi istatistiğini** öğreneceğini belirliyor.

$$\arg\min_{\hat{y}} \; \mathbb{E}\big[(y-\hat{y})^2\big] = \mathbb{E}[y \mid x]$$

Somut doğrulama — hedefler $[1, 2, 10]$ için tek bir sabit tahmin arandığında:

| Loss | En küçük yapan değer | Karşılığı |
|---|---|---|
| Kare hata (MSE) | 4.3333 | ortalama |
| Mutlak hata (MAE) | 2.0000 | medyan |

MSE ile eğitilen model koşullu **ortalamayı**, MAE ile eğitilen **medyanı**, cross-entropy ile eğitilen ise **dağılımın tamamını** tahmin etmeyi öğreniyor.

Bigram modelinde istenen şey tek bir "doğru harf" değil, harflerin dağılımıydı — NLL'in seçilmesinin sebebi bu.

---

## Neden 3 kelimeyle ölçünce de aynı sonuç çıktı

**Bu bölüm neden var:** Küçük bir örneklemle ölçüp doğru sonuca yakın çıkmak şaşırtıcı ve yanıltıcı. Neden yakın çıktığını ve **neden yine de güvenilmez olduğunu** ayırmak gerekiyor.

Kodda önce `words[0:3]` ile çalışıldı — sadece `emma`, `olivia`, `ava`. Çıkan sayı **2.4255**, tüm veriyle çıkan **2.4540**'a çok yakın. Neden?

Çünkü ölçtüğün şey **bigram başına ortalama**. Kelime sayısı arttıkça ortalama değişmiyor, sadece **daha az sapıyor**.

### Ama 3 kelime güvenilir değil — şans eseri yakın çıktı

Rastgele seçilmiş kelime gruplarıyla 400'er deneme yapıldığında:

| Kaç kelime | Ortalama tahmin | Sapma (std) | Gözlenen aralık |
|---|---|---|---|
| 3 | 2.4602 | 0.1917 | 2.040 – 3.192 |
| 10 | 2.4508 | 0.1031 | 2.111 – 2.721 |
| 50 | 2.4514 | 0.0439 | 2.323 – 2.584 |
| 200 | 2.4522 | 0.0219 | 2.396 – 2.525 |
| 1000 | 2.4537 | 0.0105 | 2.427 – 2.486 |

![Örnek büyüklüğü arttıkça tahminin daralması](/yz50/ornek-buyuklugu-nll.gif)

Üç kelimeyle ölçtüğünde sonuç 2.04 ile 3.19 arasında herhangi bir yere düşebilirdi. `emma, olivia, ava` tesadüfen tipik kelimelerdi.

**Ortalama her boyutta doğru yerde** (hepsi 2.45 civarı) — çünkü ampirik ortalama yansız bir tahmin edici. Değişen tek şey **yayılma**: örnek sayısı 100 kat artınca sapma 10 kat azalıyor. Bu, yukarıdaki $\sigma/\sqrt{n}$ kuralının aynısı.

### O zaman neden 3 kelimeyle başlanıyor

Ölçmek için değil, **kontrol etmek** için.

16 satırlık bir çıktıya bakıp "`.e` için 0.0478 mantıklı mı, `mm` gerçekten bu kadar olası mı" diye elle doğrulayabilirsin. 228 bin satıra bakamazsın.

Yani `words[0:3]` bir hata ayıklama adımı: **kod doğru mu** sorusunu cevaplıyor. **Model ne kadar iyi** sorusunun cevabı için tüm veri gerekiyor.

Modeli raporlarken her zaman `words` kullan. Kodda değişen tek şey o dilim; `n` kendiliğinden 16'dan 228.146'ya çıkıyor.

---

## Nereden okunur

Bu notun akademik karşılığı: [Jurafsky & Martin, SLP3 Bölüm 3 — N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf). Perplexity, smoothing ve dil modeli değerlendirmesinin standart referansı. Sinirsel tarafı için [Bölüm 6 — Neural Networks](https://web.stanford.edu/~jurafsky/slp3/6.pdf). Cross-entropy'nin derivative'i ve sinir ağı bağlamı için Nielsen [Bölüm 3](http://neuralnetworksanddeeplearning.com/chap3.html).

---

**Bağlantılı:** [15 · Aktivasyon + Loss Eşleşmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/) (aktivasyon-loss eşleşmesinin GLM gerekçesi) · [18 · PyTorch Broadcasting ve keepdim Tuzağı](/posts/yz50-18-broadcasting/) · [24 · Genel Tekrar — micrograd muhasebesi](/posts/yz50-24-tekrar-micrograd-muhasebesi/)
