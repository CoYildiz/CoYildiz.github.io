---
title: "07 · Neden Gradient Descent?"
published: 2026-09-29
description: "Neden kapalı form değil de iteratif iniş: istatistiksel gerekçe, maximum likelihood bağlantısı ve random reshuffling literatürünün söyledikleri."
tags:
  - YZ50
  - Gradient Descent
  - İstatistik
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

Kaynak: Nielsen Bölüm 1 + kullanıcı sorusu (2026-08-28) — "mini-batch'te rastgeleliği hangi istatistiki yönteme göre seçiyoruz" ve "neden lineer cebir/düz istatistikle kapalı-form çözüm bulmuyoruz". Bilinçli olarak ezber değil, gerekçe odaklı.

---

## Kısım 1 — Mini-Batch'teki Rastgelelik Tam Olarak Ne

**Bu kısım neden var:** Mini-batch'te örnekleri "rastgele seç" deniyor ama hangi anlamda rastgele, ve rastgeleliği bozarsan tam olarak ne kırılır? Cevap iki istatistiksel özellikte: yansızlık ve varyans. İkisini de ispatlayarak geçiyoruz, çünkü batch boyutu seçimi bunlara dayanıyor.

### Formal çerçeve

Tüm veri setini bir **popülasyon**, tek bir eğitim örneğinin gradient'inı ($\nabla C_x$) o popülasyondan çekilen bir **rastgele değişken** gibi düşün. Gerçek gradient:

$$\nabla C = \mathbb{E}_{x \sim D}[\nabla C_x] = \frac{1}{n}\sum_{i=1}^n \nabla C_{x_i}$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $\nabla C$ | Gerçek (tüm veri setine göre) gradient — her weight/bias için bir partial derivative içeren vektör |
| $\mathbb{E}$ | Beklenen değer (expectation) |
| $x \sim D$ | "x, D dağılımından rastgele çekiliyor" |
| $D$ | Eğitim verisi üzerindeki (uniform) dağılım |
| $n$ | Toplam eğitim örneği sayısı |
| $\sum_{i=1}^{n}$ | i=1'den n'e kadar toplam |
| $x_i$ | i'inci eğitim örneği |
| $\nabla C_{x_i}$ | Sadece $x_i$ örneğine göre hesaplanan gradient |

Burada D, n eğitim örneği üzerindeki (uniform) dağılım. Bu, "gradient" dediğimiz şeyin aslında bir **popülasyon ortalaması** olduğu anlamına geliyor — istatistik diliyle konuşuyoruz artık, ML diliyle değil.

### Neden yansız (unbiased) — ispat sezgisi

Mini-batch'teki her $X_j$, **aynı** D dağılımından rastgele çekiliyorsa:

$$\mathbb{E}[\nabla C_{X_j}] = \nabla C \quad \text{(her } j \text{ için ayrı ayrı doğru)}$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $X_j$ | Mini-batch'teki j'inci rastgele seçilmiş örnek — küçük harf $x_i$'den farkı: burada "hangi örneğin seçileceği" de rastgele, büyük harf istatistikte "rastgele değişken" olduğunu vurguluyor |
| $\nabla C_{X_j}$ | O örneğin gradient'i |

Beklenen değerin doğrusallığından (linearity of expectation — kalkülüsten hatırlarsın, toplamın beklenen değeri, beklenen değerlerin toplamı):

$$\mathbb{E}\left[\frac{1}{m}\sum_{j=1}^m \nabla C_{X_j}\right] = \frac{1}{m}\sum_{j=1}^m \mathbb{E}[\nabla C_{X_j}] = \frac{1}{m}\sum_{j=1}^m \nabla C = \nabla C$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $m$ | Mini-batch büyüklüğü (kaç örnek) |
| $\sum_{j=1}^{m}$ | j=1'den m'e kadar toplam (mini-batch'in içindeki örnekler üzerinden) |

Denklem soldan sağa okunuyor: "ortalamanın beklenen değeri" = "beklenen değerlerin ortalaması" (doğrusallık) = "hepsi zaten ∇C'ye eşit olduğu için ∇C'nin ortalaması" = ∇C.

Yani mini-batch'in ortalama gradient'inın **beklenen değeri, gerçek gradient'in kendisi.** Bu **rastgeleliğin zorunlu olma sebebinin matematiksel kanıtı** — ispat, $X_j$'lerin D ile **aynı dağılımdan** geldiği varsayımına dayanıyor. Rastgele olmayan bir seçim (örn. dosyadaki ilk m örnek) bu varsayımı bozar — o alt-kümenin kendi dağılımı D'den **farklı** olabilir (örn. etikete göre sıralıysa), bu durumda $E[\nabla C_{X_j}] \neq \nabla C$ olur ve tahmin **sistematik olarak yanlış** (biased) çıkar — gürültülü değil, yanlı.

### Varyans — "ne kadar rastgele/güvenilir" sorusunun cevabı

**Bu bölüm neden var:** Yansızlık, tahminin *ortalamada* doğru olduğunu söylüyor ama tek bir
mini-batch'in *o anki* tahmini hâlâ sapabilir. "Ne kadar sapar" sorusunun cevabı batch boyutunu
seçmenin tek gerekçesi — o yüzden formülü yazıp geçmek yerine nereden geldiğini çıkarıyoruz.

> **Hatırlatman gereken üç kural:**
> - $\text{Var}[aX] = a^2\,\text{Var}[X]$ — sabit çarpan **karesiyle** dışarı çıkar
> - $\text{Var}[X+Y] = \text{Var}[X] + \text{Var}[Y] + 2\,\text{Cov}(X,Y)$
> - Bağımsızlık $\Rightarrow \text{Cov} = 0$, dolayısıyla toplamın varyansı varyansların toplamı

**Türetim.** $g_j = \nabla C_{X_j}$ tek bir örneğin gradient'i olsun; $j = 1\dots m$ bağımsız ve aynı
dağılımlı, $\text{Var}[g_j] = \sigma^2$. Mini-batch tahmini $\hat{g} = \frac{1}{m}\sum_j g_j$:

$$
\text{Var}[\hat{g}]
= \text{Var}\!\left[\tfrac{1}{m}\textstyle\sum_j g_j\right]
\;\overset{(1)}{=}\; \frac{1}{m^2}\,\text{Var}\!\left[\textstyle\sum_j g_j\right]
\;\overset{(2)}{=}\; \frac{1}{m^2}\sum_j \text{Var}[g_j]
= \frac{1}{m^2}\cdot m\,\sigma^2
= \frac{\sigma^2}{m}
$$

- **(1)** birinci kural: $1/m$ sabiti kare olarak dışarı çıkıyor. Payda $m^2$ buradan.
- **(2)** ikinci ve üçüncü kural: kovaryanslar **bağımsızlık sayesinde** düşüyor.

**(2) adımı bu notun ekseni.** Eğer örnekler bağımsız seçilmeseydi (örneğin veriyi sıralı okusaydın
ve komşu örnekler benzer olsaydı), $\text{Cov} \neq 0$ olur ve varyans $\sigma^2/m$'den **büyük**
çıkardı — yani $m$'yi büyütmenin faydası düşerdi. **Rastgele seçimin zorunlu olmasının ikinci
sebebi bu**; birincisi bir önceki bölümdeki yansızlıktı. Aşağıdaki "yerine koyarak mı koymadan mı"
tartışması doğrudan bu adımın ne kadar geçerli olduğuyla ilgili.

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $\text{Var}[\cdot]$ | Varyans — bir rastgele değişkenin ortalama etrafında ne kadar yayıldığının ölçüsü |
| $\sigma^2$ | Tek bir örneğin gradient'inın popülasyon (tüm veri seti) içindeki varyansı — sabit bir sayı, veri setine ait, senin seçtiğin bir şey değil |
| $m$ | Mini-batch büyüklüğü — payda büyüdükçe varyans küçülüyor |

$\sigma^2$, tek bir örneğin gradient'inın popülasyon içindeki varyansı. Bu **standart hata $\sim 1/\sqrt{m}$** kuralı — anket örneğinde 1000 kişilik bir örneklemin hata payının (~%3), 100 kişilik örneklemden (~%10) neden çok daha düşük olduğunun aynı matematiği (Merkezi Limit Teoremi'nin doğrudan sonucu).

**Bunun pratik sonucu — neden m sürekli büyütülmüyor:** $m$'yi 4 katına çıkarmak varyansı 4'e böler ama standart sapmayı (asıl "gürültü" hissedilen şey) sadece 2'ye böler (√4=2) — yani hesaplama maliyeti **doğrusal** artarken fayda **karekök** oranında artıyor, azalan getiri. Bu yüzden pratikte "makul" bir m (32-256 gibi) seçilir, "büyüdükçe iyi" diye sonsuza gidilmez.

### Eksik Kalan Parça — $\sigma$ mu, $s$ mi? (Anket Analojisinin Tam Hali)

Yukarıdaki $\mathrm{Var} = \sigma^2/m$ formülü, $\sigma^2$'nin (gerçek popülasyon varyansı) **bilindiğini** varsayıyor — ama gerçekte **bilmiyoruz.** Anket örneğiyle devam edelim: gerçek σ (tüm ülkenin görüşündeki gerçek varyans) asla tam bilinmez, sadece **örnekten hesaplanan tahmini** s ile çalışılır:

$$s^2 = \frac{1}{m-1}\sum_{j=1}^m (\nabla C_{X_j} - \overline{\nabla C})^2$$

**Semboller:** $s^2$ = **örnek** varyansı ($\sigma^2$'nin tahmini, $\sigma^2$'nin kendisi değil) · m-1 = payda'da m değil m-1 (Bessel düzeltmesi — s^2'nin kendisinin yansız bir tahminci olması için gerekli, m yerine m-1 kullanmazsan s^2 sistematik olarak biraz küçük çıkar) · ∇C̄ = mini-batch'in ortalama gradient'i (yukarıda hesapladığımız şey).

**ML'deki asimetri:** anketin amacı "ne kadar eminiz" diye **rapor vermek** — bu yüzden s'yi hesaplayıp margin-of-error üretmek zorunlu. Mini-batch gradient descent'te ise s'yi **hiç hesaplamıyoruz** — amacımız güven aralığı raporlamak değil, sadece adım atmak. Gürültülü tahmini olduğu gibi kullanıp geçiyoruz; **çok adım tekrarlandıkça** gürültü kendiliğinden ortalanıyor (anketten farklı olarak burada "tek seferlik kesin bir cevap" değil, sürekli tekrarlanan bir süreç var).

**Ama bir yerde s'ye çok benzer bir şey var — bkz. [09](/posts/yz50-09-optimizer-varyantlari/):** Adam/RMSprop'un v(t) terimi, gradient'in **ikinci momentini** (karelerin hareketli ortalamasını) sürekli takip ediyor — bu, s^2'nin ML'deki dinamik/sürekli-güncellenen karşılığı. Adaptif optimizer'lar, "bu parametrenin gradient'i ne kadar gürültülü/değişken" sorusunun tahmini cevabını kullanarak learning rate'i parametre başına ayarlıyor — anketin s'sini hiç hesaplamayan düz SGD (Stochastic Gradient Descent)'nin aksine, Adam bunu **zımnen** hesaplıyor.

**Dağılımın şekli önemli mi?** Hayır — Merkezi Limit Teoremi tam olarak bunu garanti ediyor: $\nabla C_x$'in kendisi hangi şekilde dağılmış olursa olsun (çarpık, çok-modlu, ne olursa olsun), **mini-batch ortalamasının** dağılımı m yeterince büyüdükçe normale yaklaşıyor. Altta yatan dağılımın tam şeklini bilmemize hiç gerek yok.

### Hangi istatistiki yöntem — "yerine koyarak" mı "koymadan" mı örnekleme

İki farklı rastgele örnekleme şeması var, ayrımı önemli:

1. **Yerine koyarak (with replacement):** teorik/"ders kitabı" modeli — her $X_j$ birbirinden tamamen bağımsız, aynı örnek bir epoch içinde birden fazla kez seçilebilir, bazıları hiç seçilmeyebilir. Yukarıdaki $\text{Var} = \sigma^2/m$ formülü tam olarak bunun için geçerli.
2. **Yerine koymadan, epoch başında karıştırma (shuffle-then-partition):** pratikte gerçekte kullanılan yöntem (PyTorch `DataLoader(shuffle=True)`). İstatistiksel olarak **sonlu popülasyon düzeltmesi** (finite population correction) devreye girer:

$$\text{Var}_{\text{yerine koymadan}} = \frac{\sigma^2}{m} \cdot \frac{N-m}{N-1}$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $N$ | Popülasyon büyüklüğü (tüm veri setindeki toplam örnek sayısı — n ile aynı şey, burada istatistik literatüründeki N harfi kullanıldı) |
| $m$ | Mini-batch büyüklüğü |
| $(N-m)/(N-1)$ | "Sonlu popülasyon düzeltme çarpanı" — N sonsuz olsaydı (ya da yerine koyarak örnekleseydik) bu çarpan tam 1 olurdu, sonlu ve yerine koymadan olduğu için 1'den küçük çıkıyor |

Bu düzeltme çarpanı $\dfrac{N-m}{N-1}$ her zaman $\le 1$, yani **tek bir batch'in gradient
tahmininin varyansı** yerine koymadan örneklemede biraz daha düşük. $N$ (60.000 gibi) $m$'den
(32-256) çok büyük olduğunda çarpan ≈1'e yaklaşıyor ve fark ihmal edilebilir hale geliyor.

> **⚠️ Gerekçe zayıf — bu argümanı fazla ileri götürme** *(2026-09-20'de eklendi)*
>
> Yukarıdaki FPC hesabı **tek bir tahminin varyansı** hakkında. Pratikte "yerine koymadan"
> tercih edilmesinin sebebi olarak sunulması bir **kategori atlaması**: SGD'de asıl soru bir
> tahminin varyansı değil, **algoritmanın yakınsama hızı** — bu tamamen ayrı bir analiz.
>
> **Literatürün gerçekte söylediği:** bu yönteme **random reshuffling** (RR) deniyor — her epoch
> başında veriyi karıştırıp sırayla batch'lere bölmek, yani PyTorch'ta `DataLoader(shuffle=True)`.
> RR'nin yerine koyarak örneklemeden **genellikle** daha hızlı yakınsadığı gösterildi
> ([Mishchenko, Khaled & Richtárik, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/c8cc6e90ccbff44c9cee23611711cdc4-Paper.pdf)).
> Mekanizma FPC değil: her fonksiyonun epoch başına **tam bir kez** katkı vermesi, toplam
> yapısından (finite-sum) yararlanmayı mümkün kılıyor.
>
> **Ama koşulsuz değil.** Safran & Shamir'in makalesinin başlığı bunu doğrudan söylüyor:
> *["Random Shuffling Beats SGD Only After Many Epochs on Ill-Conditioned Problems"](https://arxiv.org/pdf/2106.06880)* —
> kötü koşullanmış problemlerde üstünlük ancak **çok sayıda epoch'tan sonra** ortaya çıkıyor.
> Yani "asla daha kötü değil" ifadesi fazla güçlü; doğrusu **"tipik olarak daha iyi, her koşulda
> değil."**

Yerine koymadan örneklemenin tartışmasız avantajı ise şu: her örneğin epoch başına **tam bir kez**
kullanılmasını garanti ediyor — yerine koyarak örneklemede bazı örnekler hiç görülmezken bazıları
birkaç kez geliyor.

3. **Stratified/class-balanced (bkz. [08](/posts/yz50-08-stochastic-gradient-descent/)):** saf rastgeleliğin bir kaynağını (sınıf dağılımı) kasıtlı olarak sabitleyip geri kalanını rastgele bırakarak varyansı daha da azaltan bir teknik — istatistikte buna **stratified sampling** deniyor, anket araştırmalarında da (yaş/cinsiyet gruplarına göre kota koyup içeride rastgele seçmek) birebir aynı mantıkla kullanılıyor.

---

## Kısım 2 — Neden Lineer Cebirle Kapalı-Form Çözüm Yok

**Bu kısım neden var:** Lineer regresyonda $\nabla C = 0$ denklemini çözüp tek adımda cevaba varabiliyorduk. Sinir ağında neden varamıyoruz — tembellikten mi, yoksa ispatlanmış bir imkânsızlıktan mı? Bu ayrım önemli, çünkü birincisi "daha iyi bir yöntem aranabilir" demek, ikincisi "aramak boşuna" demek.

### Önce iyi haber — lineer regresyonda VAR

Sezgin yanlış değil. Model **doğrusal** olduğunda (ŷ = Xw) ve maliyet **kareler toplamı** olduğunda (C = ‖y-Xw‖^2), gerçekten kapalı-form bir çözüm var — buna **Normal Denklem** deniyor:

$$C(w) = (y-Xw)^T(y-Xw)$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $C(w)$ | w'ye bağlı maliyet fonksiyonu |
| $y$ | Gerçek/hedef değerlerin vektörü (etiketler) |
| $X$ | Girdi verisi matrix'i (her satır bir eğitim örneği, her sütun bir özellik/feature) |
| $w$ | Model weight'lerinın vektörü |
| $Xw$ | Modelin tahminleri ($\hat{y}$) |
| $(\cdot)^T$ | Transpoze — bir matrix'in satır ve sütunlarını yer değiştirme, iki vektörün "iç çarpımını" matrix çarpımı olarak yazabilmek için gerekli |

Gradient'i al, sıfıra eşitle:

$$\nabla_w C = -2X^T(y-Xw) = 0 \implies X^TXw = X^Ty \implies w^* = (X^TX)^{-1}X^Ty$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| $\nabla_w C$ | C'nin w'ye göre gradient'i — alt simge w, "hangi değişkene göre derivative aldığımızı" belirtiyor |
| $\implies$ | "Buradan şu çıkar" (mantıksal sonuç oku) |
| $X^TX, X^Ty$ | X'in transpozu ile yapılan matrix çarpımları |
| $(X^TX)^{-1}$ | $X^TX$ matrix'inin tersi (matrix'te "bölme" karşılığı) |
| $w^*$ | Optimal weight vektörü — yıldız işareti burada "bulduğumuz en iyi/nihai değer" anlamında, üs değil |

Bu gerçekten kullanılıyor — `sklearn.LinearRegression`, küçük/orta veri setlerinde tam olarak bunu hesaplıyor, gradient descent'e hiç gerek duymadan. **Neden işe yarıyor:** C(w), w'ye göre **kuadratik** bir fonksiyon (tek, temiz bir çanak/parabol şekli — tek global minimum, başka hiçbir durağan nokta yok). Gradient'i sıfıra eşitlemek, w için **doğrusal bir denklem sistemi** veriyor — ve doğrusal sistemler matrix tersiyle **tam olarak** çözülebiliyor.

### Neden sinir ağında bu çöküyor

Bir sinir ağında (senin `Neuron`'un dahil), çıktı **nonlineer bir aktivasyon** içeriyor: $\hat{y} = \sigma(Xw)$. C(w) = ‖y - σ(Xw)‖^2 artık w'nin kuadratik bir fonksiyonu **değil** — σ'nın doğrusal-olmayanlığı bunu bozuyor. ∇C = 0'ı çözmeye çalıştığında, artık nonlineer, **nonlineer bir denklem sistemi** elde ediyorsun — ve nonlineer sistemlerin genel olarak kapalı-form (cebirsel) çözümü **yok**.

**Bunun ne kadar temel bir sınır olduğunu hissetmen için (Abel-Ruffini paraleli):** cebirden hatırlarsın — 5. dereceden ve üstü polinom denklemlerin **genel bir radikal (kapalı-form) çözüm formülü olmadığı ispatlanmış** (Abel-Ruffini teoremi). Bu, "biz henüz bulamadık" değil, **ispatlanmış olarak var olmadığı** anlamına geliyor. Sinir ağının ∇C=0 koşulu, layer'lar arası matrix çarpımları + nonlineer aktivasyonların iç içe geçmesinden doğan, çok daha karmaşık bir nonlineer denklem sistemi — Abel-Ruffini ile birebir aynı problem değil ama **aynı ailenin bir üyesi**: nonlineer denklemler için kapalı-form çözüm bulmak istisna, kural değil. Bu yüzden **sayısal/iteratif** yöntemlere (gradient descent) mecbursun — kapalı-form arayışı prensipte bile boşuna.

**Ekstra sorun — convexity (convexity):** lineer regresyonun çanağı **tek** minimuma sahip. Sinir ağının maliyet yüzeyi genelde **convex değil** (non-convex) — birden çok local minimum, saddle point var. Yani ∇C=0'ı bir şekilde sayısal olarak çözebilsen bile, "tek bir doğru cevap" diye bir şey yok, hangi durağan noktaya düştüğün başlangıç noktana bağlı.

![Convex vs non-convex maliyet yüzeyi](/yz50/convex-vs-nonconvex.svg)

**Gerçek dünyadan bir veri noktası:** logistic regression (senin gördüğün gibi tek layer + sigmoid) de kapalı-form çözüme sahip değil — ama **convex** olduğu için (tek minimum garantili) ve boyutu genelde küçük olduğu için, pratikte genelde **Newton's method/IRLS** ile çözülüyor — yani tam olarak birkaç mesaj önce konuştuğumuz Hessian'ı (second derivative'i) kullanan yöntem. Sinir ağında bunun neden mümkün olmadığını zaten biliyorsun: milyonlarca parametrede Hessian O(n^2), hesaplanamaz kadar pahalı. Logistic regression küçük olduğu için bu maliyeti karşılayabiliyor, derin ağ karşılayamıyor.

---

## Kısım 3 — Neden "Düpedüz İstatistik" (MLE, Rastgele Arama) da Değil

**Bu kısım neden var:** Kapalı form yoksa akla gelen iki alternatif var: istatistiğin klasik aracı MLE, ve en kaba yöntem olan rastgele arama. İkisinin de neden çözüm olmadığını göstermeden "gradient descent şart" demek eksik kalır.

### MLE de aynı duvara çarpıyor

Maximum Likelihood Estimation (istatistikte "en iyi" parametreyi bulmanın klasik yolu) bazı özel durumlarda kapalı-form veriyor (lineer regresyonun MLE'si, Gauss gürültü varsayımı altında, yukarıdaki Normal Denklem'in ta kendisi). Ama genel olarak MLE de **log-likelihood'un gradient'inı sıfıra eşitleyip çözmeyi** gerektiriyor — model nonlineer bir şey içerdiği anda **aynı** nonlineer denklem sistemi problemine çarpıyorsun. MLE, gradient descent'in **alternatifi değil** — çoğu zaman MLE'yi sayısal olarak çözmenin **aracı bizzat gradient descent'in kendisi.**

### Rastgele arama neden işe yaramaz — boyutun laneti

Diyelim istatistiksel bir yaklaşım denedin: rastgele binlerce weight kombinasyonu dene, en düşük maliyetliyi seç. Hafta 1'den hatırla — o basit ağın bile **13.002 parametresi** vardı. Her parametrenin "iyi" aralığının cömertçe **%1**'lik bir dilim olduğunu varsaysak bile, rastgele bir noktanın **13.002 boyutun hepsinde aynı anda** o dilime düşme ihtimali:

$$(0.01)^{13002}$$

**Semboller:**

| Sembol | Anlamı |
|---|---|
| 0.01 | Her bir parametrenin "iyi" aralığının, o parametrenin toplam olası aralığına oranı (%1 — cömert bir varsayım) |
| üs (13002) | Boyut/parametre sayısı — üste çıkmasının sebebi, "hepsinin AYNI ANDA iyi olması" gerekliliği (bağımsız probability'lerin çarpımı, probability/istatistikten hatırlarsın: P(A ve B) = P(A)·P(B)) |

**Bu sayıyı hissetmek için:** $(0.01)^{13002} = 10^{-26004}$. Gözlemlenebilir evrendeki atom
sayısı $\approx 10^{80}$, evrenin yaşı $\approx 10^{17}$ saniye. Saniyede bir atom kadar deneme
yapsan bile $10^{97}$ deneme eder — gereken sayının yanında yuvarlama hatası. Buna **boyutun
laneti** (curse of dimensionality) deniyor: "iyi" bölgenin toplam hacme oranı boyutla
**katlanarak** küçülüyor.

**Dikkat, buradaki asıl ders "rastgele arama yavaş" değil.** Rastgele arama boyuttan bağımsız
olarak aynı hızda çalışıyor; değişen şey **aradığı şeyin ne kadar küçüldüğü**. Gradient descent
ise hacmi taramıyor, yerel eğimi izliyor — maliyeti boyutla doğrusal artıyor, üstel değil. Fark
hız farkı değil, **cins farkı**.

**Gradient descent'in gerçek avantajı burada ortaya çıkıyor:** kör arama yapmıyor — her noktada, o noktanın **yerel eğimine** bakıp doğrudan iyileşme yönünü biliyor. Rastgele aramanın aksine, her adımda "nereye gideceğini" tahmin etmesi gerekmiyor, gradient zaten söylüyor. Bu yüzden gradient descent, sıfırıncı-mertebe (gradientsız, kör arama gibi — çok verimsiz) ile ikinci-mertebe (Hessian kullanan, çok pahalı) yöntemler arasında bilinçli bir orta yol: **yeterince bilgili (yerel eğimi kullanıyor) ama yeterince ucuz (sadece birinci derivative).**

---

## Özet — Üç Sorunun Tek Ortak Kökü

1. Mini-batch'te rastgelelik şart çünkü **yansızlık** rastgeleliğe dayanıyor (istatistik) — rastgele olmayan seçim sistematik hata (bias) getirir.
2. Kapalı-form yok çünkü model **nonlineer** → maliyet **kuadratik değil** → ∇C=0 **nonlineer bir sistem**, genel olarak cebirsel çözümü yok (Abel-Ruffini'nin ruhu).
3. Rastgele arama/MLE de yardımcı olmuz çünkü **boyutun laneti** — yüksek boyutta kör arama imkansız, MLE de aynı nonlineer duvara çarpıyor.

**Ortak payda:** gradient descent, "kapalı-form yok, kör arama da imkansız" boşluğunu dolduran, yerel bilgiyi (gradient'i) kullanan genel-amaçlı sayısal yöntem. Zarif değil ama **çalışıyor** — ve nöral ağların neredeyse tamamının (senin `Value`/`Neuron`'un dahil) üzerine kurulduğu şey bu.

---

## Bağlantı

Bu not, [08](/posts/yz50-08-stochastic-gradient-descent/) (mini-batch pratikleri) ve [09](/posts/yz50-09-optimizer-varyantlari/) (Hessian'dan kaçış) notlarının **neden** sorusuna cevap veriyor — o notlar "ne yapılıyor", bu not "neden başka türlü yapılamıyor".

**Bağlantılı:** [08 · Stochastic Gradient Descent](/posts/yz50-08-stochastic-gradient-descent/) · [09 · Optimizer Varyantları — Momentum, AdaGrad, RMSprop, Adam](/posts/yz50-09-optimizer-varyantlari/).
