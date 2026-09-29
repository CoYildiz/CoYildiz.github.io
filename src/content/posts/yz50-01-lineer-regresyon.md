---
title: "Lineer Regresyon"
published: 2026-09-29
description: "İki parametreli en küçük modelden başlayıp loss, gradient ve öğrenme oranının nereden geldiğini türetiyoruz. Learning rate'in üst sınırı neden 2/L?"
tags:
  - YZ50
  - Lineer Regresyon
  - Optimizasyon
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

En küçük model: tek girdi, tek çıktı, iki parametre. Buradaki her parça daha sonra olduğu gibi
ölçekleniyor — parametre, loss, gradient, öğrenme. Üç ayrı çözüm yolu var ve üçünün
karşılaştırması, sinir ağlarında neden yalnızca birinin kaldığını açıklıyor.

### Bu notta kullanılan veri

Perrotta'nın kitabındaki oyuncak veri seti ([`more_content/data/pizza.txt`](https://github.com/CoYildiz/yz50/blob/main/more_content/data/pizza.txt)). **Bir pizzacının
günlük kaydı:** o gün kaç rezervasyon alındığı, ve o gün kaç pizza satıldığı.

| Reservations (girdi $x$) | Pizzas (çıktı $y$) |
|---|---|
| 13 | 33 |
| 2 | 16 |
| 14 | 32 |
| ... | ... |

**30 satır, tek girdi, tek çıktı.** Soru: *"bugün 20 rezervasyon aldım, kaç pizza hazırlamalıyım?"*
Aradığın şey bu iki sütun arasındaki doğru — `pizza ≈ w × rezervasyon + b`.

Veri bilinçli olarak oyuncak: 30 nokta ve tek değişken, her şeyi elle hesaplayıp grafiğini
çizebileceğin kadar küçük. Aşağıdaki bütün sayılar bu veriden hesaplandı.

---

## 1. Model

**Bu bölüm neden var:** "lineer" kelimesinin neyi nitelediği sürekli yanlış anlaşılıyor, ve bu yanlış anlama sinir ağlarının nerede başladığını gizliyor.

$$\hat{y} = wx + b$$

**"Lineer" olan girdi değil, parametreler.** $\hat{y} = w x^2 + b$ de lineer regresyondur —
$x^2$'yi bir özellik olarak alırsın, model hâlâ $w$ ve $b$'de lineerdir ve aşağıdaki kapalı form
aynen çalışır. Lineerliği bozan şey $\hat{y} = \tanh(wx + b)$ gibi parametreyi
nonlineer bir fonksiyonun içine sokmak. Sinir ağlarını ayıran çizgi tam burası.

Loss olarak ortalama kare hata:

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}\left(wx_i + b - y_i\right)^2$$

---

## 2. Üç çözüm yolu ve veri büyüdükçe ne olduğu

**Bu bölüm neden var:** Aynı minimumu bulan üç yöntem var ve üçü de bu veride çalışıyor. Ama
30 örnekli, 1 özellikli bir problemde hepsi çalışır — ayırt edici soru bu değil. Asıl soru
**veri büyüdükçe hangisinin ayakta kaldığı**, çünkü sinir ağlarına taşınan yöntemi o belirledi.

### Bu veride sonuçlar

Veri: $n = 30$ örnek, $d = 1$ özellik (rezervasyon), 2 parametre.

| yöntem | $w$ | $b$ | loss | maliyet |
|---|---|---|---|---|
| Koordinat araması (tepe tırmanışı) | 1.1000 | 12.930 | 22.8636 | 1551 iterasyon, 18.6 ms |
| Gradient descent (lr=0.001) | 1.0811 | 13.172 | 22.8427 | 20.000 iterasyon |
| Kapalı form (OLS — en küçük kareler) | 1.0811 | 13.173 | 22.8427 | tek geçiş, 0.013 ms |

Gradient descent ve kapalı form **aynı** cevaba varıyor (4 hane). Koordinat araması biraz uzakta —
sebebi aşağıda. Bu ölçekte üçü arasında pratik bir fark yok.

### Ölçekleme — asıl soru bu

Sentetik regresyon verisinde $d$ (özellik sayısı) ve $n$ (örnek sayısı) değiştirilip her yöntemin
süresi ölçüldü:

![Üç yöntemin ölçeklenmesi](/yz50/uc-yontem-olcekleme.png)

**Ölçülen büyüme üsleri** (log-log eğimi; üs 1 = doğrusal, 2 = kareyle, 3 = küple):

| yöntem            | $d$ ile büyüme üssü | neden                                                                                                           |
| ----------------- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| kapalı form       | **1.1**             | Bu aralıkta $n \gg d$, yani $X^\top X$'i kurmanın $\mathcal{O}(nd^2)$ maliyeti değil BLAS (NumPy'ın altındaki optimize lineer cebir kütüphanesi)'ın verimliliği baskın |
| gradient descent  | **0.8**             | adım başına $\mathcal{O}(nd)$, ama vektörleştirme sayesinde küçük $d$'de sabit gider baskın                     |
| koordinat araması | **1.4**             | adım başına $2d$ ayrı loss hesabı, her biri $\mathcal{O}(nd)$                                                   |

**Ölçümün söylediği, beklediğimin tersi:** kapalı form denenen **bütün** boyutlarda ($d = 2$'den
$d = 6400$'e) en hızlısı. Süre bakımından bir kesişim **yok**. `np.linalg.solve` LAPACK/BLAS
kullanıyor ve bu boyutlarda 200 adımlık bir GD döngüsünü rahatça geçiyor.

> **Düzeltme (2026-09-20):** Bu notun önceki hali "birkaç bin özelliği geçince iteratif yöntem
> kazanır" diyordu — **yanlış gerekçe.** Kesişim $d$'de değil, aşağıda görüleceği gibi $n$'de ve
> farklı bir GD türünde.

### Ama gerçekte gradient descent kazanıyor — nerede?

Yukarıdaki ölçüm bir şeyi kaçırıyor: **tam-batch GD'yi sabit adım sayısıyla** ölçtü. Pratikte
kullanılan şey o değil, **SGD (Stochastic Gradient Descent)** — her adımda verinin küçük bir rastgele parçasına bakan hali. Ve
asıl fark $d$'de değil $n$'de ortaya çıkıyor.

Doğru karşılaştırma: *"aynı loss'a varmak ne kadar sürüyor?"* ($d = 100$ sabit, hedef = kapalı
formun loss'unun %1 üstü):

| $n$ | kapalı form | SGD | oran | SGD verinin ne kadarını gördü |
|---|---|---|---|---|
| 500.000 | 0.10 s | 0.02 s | **4.2x** | %20.5 |
| 2.000.000 | 0.38 s | 0.05 s | **6.9x** | %5.1 |
| 6.000.000 | 1.35 s | 0.13 s | **10.3x** | %1.7 |

**Son sütun asıl cevabı veriyor.** Kapalı form $X^\top X$'i kurmak için veri setinin **tamamını**
görmek zorunda — maliyeti $n$ ile doğrusal artıyor. SGD ise belirli bir doğruluğa varmak için
kabaca **sabit sayıda örnek** görüyor; bu sayı verinin büyüklüğünden bağımsız, gürültü ve
condition number'dan geliyor. Veri 12 kat büyüdüğünde SGD'nin gördüğü oran %20'den %1.7'ye
düşüyor, süresi neredeyse aynı kalıyor.

Yani **oran $n$ ile büyüyor ve üst sınırı yok.** Gerçek veri setlerinde $n$ milyonlar ya da
milyarlar olduğu için pratikte GD/SGD'nin kazanması buradan geliyor — küp maliyetten değil.

> **Özet — üç ayrı rejim:**

| Durum                | Kazanan         | Sebep                                                              |
| -------------------- | --------------- | ------------------------------------------------------------------ |
| küçük $n$, küçük $d$ | **kapalı form** | tek geçişte kesin cevap; GD kullanmak gereksiz                     |
| büyük $n$            | **SGD**         | sabit sayıda örnekle yetiniyor, kapalı form hepsini görmek zorunda |
| büyük $d$            | **GD/SGD**      | $d\times d$ matrix belleğe sığmıyor                                |
| nonlineer model      | **GD/SGD**      | kapalı form **yok**                                                |

### Kapalı formu bitiren şey süre değil, üç başka şey

**1. Bellek — asıl duvar.** $X^\top X$ matrix'i $d \times d$. Sağdaki panelde:

| $d$ | $X^\top X$ belleği |
|---|---|
| 10.000 | 0.8 GB |
| 50.000 | 20 GB |
| 100.000 | 80 GB |

Gradient descent ise yalnızca $d$ uzunluğunda bir vektör tutuyor — $d = 100.000$'de 0.8 MB.
**Aradaki fark $10^5$ kat.** MNIST'te $d = 784$ sorun değil ama bir dil modelinde $d$ milyarlara
çıkıyor; orada $d \times d$ bir matrix diye bir şey yok.

**2. Bütün veriyi aynı anda istemesi.** Kapalı form $X^\top X$'i kurmak için veri setinin
tamamını görmek zorunda. Gradient descent bir seferde bir mini-batch ile çalışabiliyor — veri
belleğe sığmadığında ya da akış halinde geldiğinde tek seçenek bu.

**3. Koşullanmayı kareye çıkarması.** $X^\top X$ oluşturmak condition number'ı **karesine** çıkarıyor:
$\kappa(X^\top X) = \kappa(X)^2$. Bu notun 4. bölümünde $\kappa = 825.9$ görüyorsun; kapalı formda
efektif olarak $825.9^2 \approx 682.000$ ile uğraşıyorsun. Sayısal olarak kırılgan — bu yüzden
pratikte normal denklemler yerine QR ya da SVD tabanlı çözücüler tercih ediliyor.

**Ve dördüncüsü, en önemlisi:** yukarıdakilerin hepsi ölçek sorunu. Sinir ağlarında kapalı formun
terk edilme sebebi ölçek değil — 3. bölümde göreceğin gibi, nonlineer bir model için
**öyle bir formül yok.**

### Koordinat araması — neden ölçeklenmiyor

İlk yaklaşım şuydu: her adımda $w$'yi $\pm\text{lr}$ oynat, loss düştü mü bak; düşmediyse
$b$'yi oynat. Adım boyu sabit, gradient hiç hesaplanmıyor.

Çalışıyor ama **her adımda her parametre için ayrı bir loss hesabı** gerektiriyor. İki
parametre → adım başına 4 deneme. Bir sinir ağında 13.002 parametre → adım başına 26.004 ileri
geçiş.

Bu, numerical derivative'in sorununun birebir aynısı: doğru yönü bulmak için parametre sayısı kadar
sondaj yapmak.

**Bu problem çözülüyor, ama burada değil.** Çözümün adı **backpropagation**: chain rule'u ağın
grafiği üzerinde ters yönde uygulayıp bütün partial derivative'leri **tek bir geri geçişte** almak.
Maliyet parametre sayısından bağımsız hale geliyor. Nasıl yapıldığı
[11-backpropagation-sezgisi](/posts/yz50-11-backpropagation-sezgisi/) ve
[12-backpropagation-calculus](/posts/yz50-12-backpropagation-calculus/)'ta; elle kurulmuş hali
[24-tekrar-micrograd-muhasebesi](/posts/yz50-24-tekrar-micrograd-muhasebesi/)'nde.

![Koordinat araması parametre sayısıyla ölçeklenmiyor](/yz50/koordinat-aramasi-olceklenmiyor.png)

Ayrıca durma noktası da kaba: algoritma "dört tek-eksen hareketinin hiçbiri iyileştirmiyor"
dediğinde duruyor, bu "minimumdayım" demek değil — çapraz bir hareket ($w$'yi azalt + $b$'yi
artır) hâlâ iyileştirebilir. Bu yüzden $w$ 1.10'da takılıyor, gerçek optimum 1.0811.

---

## 3. Kapalı form: normal denklemler

**Bu bölüm neden var:** Lineer regresyonun tek adımda çözülen bir formülü var. Sinir ağlarında böyle bir şey yok — bu bölüm o formülü çıkarıp **neden genellenemediğini** gösteriyor.

$L$ konveks ve her yerde differentiable, yani minimum tam olarak iki partial derivative'in sıfırlandığı
yerde:

$$
\frac{\partial L}{\partial w} = \frac{2}{n}\sum x_i(wx_i + b - y_i) = 0, \qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum (wx_i + b - y_i) = 0
$$

İkinci denklemden $b = \bar{y} - w\bar{x}$; yerine koyup çözünce:

$$w = \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(X)}$$

İkinci denklemin doğrudan söylediği şey: **artıkların ortalaması sıfır**, yani regresyon doğrusu
her zaman $(\bar{x}, \bar{y})$ noktasından geçer.

### Toplam formu ↔ ortalama formu — buradaki tuzak

Ders kitaplarındaki yaygın hali toplamlarla yazılır:

$$w = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - \left(\sum x\right)^2}$$

$\sum x = n\bar{x}$ koyarak çevirince pay $n^2(\overline{xy} - \bar{x}\bar{y})$, payda
$n^2(\overline{x^2} - \bar{x}^2)$ oluyor; $n^2$ sadeleşiyor. **Ortalama formunda hiç $n$ yok.**

$n$'ler ya her iki terime uygulanır ya hiçbirine. Karıştırıp `n*mean(X*Y) - mean(X)*mean(Y)`
yazmak eğimi 1.081 yerine 1.837 veriyor — hata vermiyor, sessizce yanlış model üretiyor.
Doğrulaması ucuz: `np.polyfit(X, Y, 1)`.

### Matrix hali ve neden sinir ağlarında yok

Çok özellikli hali:

$$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$$

İki sebeple genel çözüm değil:

1. **Ölçek:** Süre değil **bellek** ve **sayısal kararlılık** — 2. bölümde ölçülerek
   gösterildi: $X^\top X$ matrix'i $d \times d$ olduğu için $d$ büyüdükçe belleğe sığmıyor, ve
   $\kappa(X^\top X) = \kappa(X)^2$ olduğu için koşullanma kötüleşiyor.
2. **Varlık:** Derivative sıfırlama denklemi ancak model parametrelerde lineerken kapalı biçimde
   çözülebiliyor. `tanh` bir nöron eklediğin anda $\partial L/\partial w = 0$ denklemi hem
   aşkın (transandantal) hem de çok bilinmeyenli hale geliyor ve **kapalı bir çözümü yok** —
   polinom denklemlerinde beşinci dereceden sonra kök formülü olmamasıyla (Abel-Ruffini) aynı
   türden bir imkânsızlık, detayı
   [07-neden-gradient-descent](/posts/yz50-07-neden-gradient-descent/)'te.

   **Bu problem nasıl aşılıyor:** kapalı çözüm aranmaktan vazgeçilip **iteratif** çözüme
   geçiliyor. Denklemi çözmek yerine minimuma adım adım yaklaşıyorsun — gradient descent tam
   olarak bunun için var. Bedeli: tek geçişte bitmiyor, ve bu notun 4. bölümündeki learning rate
   / koşullanma sorunları ortaya çıkıyor.

Yani gradient descent lineer regresyonda bir tercih, sinir ağlarında tek seçenek.

---

## 4. Learning rate'in üst sınırı nereden geliyor

> **Kapsam uyarısı:** Aşağıdaki `0.01` ve `0.004555` sayıları **bu veriye özgü** —
> [`more_content/data/pizza.txt`](https://github.com/CoYildiz/yz50/blob/main/more_content/data/pizza.txt), 30 örnek, tek özellik, standartlaştırılmamış. Başka bir veri
> setinde sınır bambaşka çıkar; `0.01` kendi başına "kötü learning rate" değil. Genel olan şey
> sayı değil, **sayıyı veriden hesaplama yöntemi.**

**Bu bölüm neden var:** Bu veride `lr = 0.01` koyunca loss birkaç yüz adımda `inf`, sonra `nan`
oluyor; `lr = 0.004` koyunca sorunsuz yakınsıyor. Aradaki sınırın keyfi olmadığını, **verinin
kendisinden hesaplanabildiğini** göstermek için. Aynı hesap ileride Kaiming init ve BatchNorm'un
gerekçesi olacak.

### "Kararlılık" burada ne demek

Gradient descent bir **özyineleme**: $w_{k+1} = w_k - \mathrm{lr}\,\nabla L(w_k)$. Yani her adımda
bir önceki noktadan yeni bir nokta üretiyorsun — zamanla ilerleyen bir sistem.

Minimum noktası bu sistemin **sabit noktası**: oraya varırsan orada kalırsın, çünkü orada
$\nabla L = 0$ ve güncelleme hiçbir şey değiştirmiyor.

Asıl soru şu: **sabit noktanın yakınında başlarsan ne oluyor?**

| Davranış | Adı |
|---|---|
| Sapma her adımda küçülüyor, sabit noktaya yaklaşıyorsun | **kararlı** (stable) |
| Sapma her adımda büyüyor, uzaklaşıyorsun | **kararsız** (unstable) — patlama |

**"Kararlılık analizi" = bu ikisini ayıran koşulu bulmak.** Terim dinamik sistemler ve kontrol
teorisinden geliyor; oradaki soru da aynı: bir denge noktası küçük sapmaları söndürüyor mu,
büyütüyor mu? Aşağıdaki 4.3'te yaptığımız tam olarak bu — sapmanın her adımda hangi sayıyla
çarpıldığını bulup o sayının 1'den küçük olma koşulunu yazıyoruz.

### Bu sonuç bana ait değil — genel hali ve nerede geçtiği

Bulacağımız $\mathrm{lr} < 2/\lambda_{\max}$ sınırı, daha genel bir teoremin bu probleme
özel hali. Genel teoremin ifadesi şöyle:

> Bir fonksiyon **$L$-smooth** ise, gradient descent $\mathrm{lr} < 2/L$ için kararlıdır.

**$L$-smooth ne demek** — tek cümleyle: *gradient hiçbir yerde belirli bir hızdan daha hızlı
değişmiyor.* Formal hali:

$$\lVert \nabla L(a) - \nabla L(b) \rVert \le \underbrace{L}_{\text{L-smooth sabiti}}\,\lVert a - b \rVert$$

> **Gösterim uyarısı:** Buradaki $L$ **loss fonksiyonu değil**, düzgünlük sabitidir
> ("$L$-smooth" ifadesindeki $L$). Literatürde ikisi de $L$ ile gösterildiği için karışıyor —
> bu notta loss $L(w,b)$ biçiminde argümanlarıyla, düzgünlük sabiti ise çıplak $L$ olarak geçiyor.

Yani iki nokta arasındaki gradient farkı, aralarındaki mesafenin $L$ katını geçemiyor. Eğrilik
yukarıdan $L$ ile sınırlı demek.

**Neden böyle bir sınır gerekiyor:** adım atarken gradient'e güveniyorsun, ama gradient attığın
adım boyunca değişiyor. Değişim hızı sınırsız olsaydı, adımın sonunda gradient tamamen başka bir
şey olabilirdi ve adım boyu hakkında hiçbir garanti veremezdin. $L$, "gradient en kötü ihtimalle
bu hızda değişir" sözü.

**Bizim durumumuzda $L = \lambda_{\max}$**, çünkü kuadratik bir loss'ta eğrilik doğrudan Hessian
tarafından veriliyor ve en büyük eğrilik $\lambda_{\max}$ (4.2'de $H$'ın sabit olması sayesinde bu
"en kötü ihtimal" her yerde aynı). Genel teoremi bu değerle yazınca $\mathrm{lr} < 2/\lambda_{\max}$
çıkıyor.

*Genel halini okumak istersen: Boyd & Vandenberghe, Convex Optimization, Bölüm 9; ya da Nesterov,
Introductory Lectures on Convex Optimization. Aşağıdaki türetim aynı sonucun, kitap açmadan
izlenebilen iki parametreli hali.*

![Kritik sınırın altında yakınsıyor, üstünde patlıyor](/yz50/kosullanma-lr-patlamasi.png)

### 4.1 Gradient yönü söylüyor, eğrilik ne kadar gidilebileceğini

Gradient sana hangi yöne gideceğini söylüyor ama **ne kadar** gidebileceğini söylemiyor. O bilgi
second derivativede: gradient'in kendisi ne hızla değişiyor. Tek değişkenli halde bu bir sayı; iki
parametrede bir $2\times 2$ matrix — **Hessian**.

Dört second derivative'i doğrudan alalım. $L(w,b) = \frac{1}{n}\sum (wx_i + b - y_i)^2$ için:

$$
\frac{\partial^2 L}{\partial w^2} = \frac{2}{n}\sum x_i^2 = 2\,\overline{x^2}, \qquad
\frac{\partial^2 L}{\partial w \partial b} = \frac{2}{n}\sum x_i = 2\bar{x}, \qquad
\frac{\partial^2 L}{\partial b^2} = 2
$$

$$H = 2\begin{bmatrix} \overline{x^2} & \bar{x} \\ \bar{x} & 1 \end{bmatrix}$$

### 4.2 Bu matrix **neye göre** sabit

Yukarıdaki ifadeye bak: içinde $w$ yok, $b$ yok, hatta $y$ bile yok. **Yalnızca girdilere bağlı.**

Bunun anlamı **parametre uzayında sabit olması**: eğitim sırasında $(w, b)$ nereye giderse gitsin
eğrilik değişmiyor. Loss bir paraboloid ve çanağın şekli her noktada aynı.

**Veriye göre sabit değil** — $X$'i değiştirirsen ($\overline{x^2}$ büyürse) $H$ de değişir. Zaten
4.4'teki standartlaştırma argümanı tam olarak buna dayanıyor.

> Kısacası: $H$, **parametrelerin değil verinin** bir fonksiyonu. Bu ayrım bu bölümün eksenidir —
> "sabit" derken kastedilen budur.

Neden önemli: eğrilik parametreye bağlı olmadığı için aşağıdaki kararlılık sınırı **bir kez**,
eğitim başlamadan önce hesaplanabiliyor ve her yerde geçerli. Genel (konveks olmayan) bir problemde
Hessian $w$ ile değişir, dolayısıyla böyle bir sınır yalnızca yerel olur ve yol boyunca yeniden
hesaplanması gerekir. Lineer regresyonun özel olduğu nokta burası.

### 4.2b Bunu kodda nerede yazdık — ve ne zaman yazmak gerekir

**Kısa cevap: hiçbir yerde yazmadın, ve yazman da gerekmiyor.** Ama $L$'nin etkisi kendi kodunun
beş ayrı yerinde duruyor. Sırayla:

| Kodda | Nerede | $L$ ile ilişkisi |
|---|---|---|
| `lr=0.001` | `A_basic_lin_reg.py`, `B_more_dimension.py`, `D_discern_mach.py` | Doğrudan $2/L$ tahmini — deneyerek bulundu |
| `lr=1e-5` | `E_mnist_classfier_for5.py` | **$L$'nin en çıplak görüldüğü yer.** Piksel değerleri 0-255 olduğu için $\overline{x^2}$ devasa, dolayısıyla $L$ devasa, dolayısıyla lr minicik olmak zorunda |
| `lrs = 10**torch.linspace(-3, 0, 1000)` | `mlp.py:89` | **$L$'yi hesaplamak yerine arıyorsun.** Learning rate taraması, $2/L$ sınırını ampirik olarak bulmanın adı |
| `W1 = randn(...) * (5/3) / fan_in**0.5` | `mlp-2.py:80` | Kaiming init — $L$'yi **başlangıçta** küçük tutma |
| `BatchNorm1d` | `mlp-2.py:207` | $L$'yi **eğitim boyunca** küçük tutma |
| `lr = 0.1 if i < 100000 else 0.01` | `mlp.py:115`, `mlp-2.py:131` | Learning rate decay — $L$'nin sabit olmadığı, eğitim ilerledikçe değiştiği varsayımı |

### Ne zaman $L$'yi gerçekten hesaplamak gerekir

**Neredeyse hiç.** Üç sebeple:

1. **Küçük, kuadratik problemde hesaplanabilir** (bu notun 4.3'ünde yaptığımız gibi, iki eigenvalue)
   — ama orada bile lr taraması daha hızlı ve daha az düşündürücü. $L$'yi hesaplamak **öğretici**,
   pratik değil.
2. **Derin ağda hesaplanamaz.** $L = \lambda_{\max}(H)$ ve $H$ matrix'i $d \times d$; $d$ milyonlarsa
   o matrix'i kurmak imkânsız. (Power iteration ile $\lambda_{\max}$ tahmin edilebilir, ama pahalı
   ve araştırma dışında yapılmıyor.)
3. **Derin ağda tek bir $L$ zaten yok.** Loss nonconvex olduğu için eğrilik $w$ ile değişiyor —
   4.2'de gördüğün "Hessian parametreye bağlı değil" güzelliği orada yok. Bir noktada hesapladığın
   $L$, on adım sonra geçersiz.

### O zaman $L$ ne işe yarıyor

**Hesaplamıyorsun — mühendisliğini yapıyorsun.** Derin öğrenmedeki bir sürü teknik, "$L$'yi
bilmiyorum ama küçük ve kontrollü tutayım" stratejisinin farklı biçimleri:

| Teknik | Ne yapıyor |
|---|---|
| Girdiyi standartlaştırmak | $L$'yi doğrudan küçültüyor (bu notun 4.5'i) |
| Kaiming / Xavier init | Aktivasyon ölçeğini kontrol altına alarak $L$'yi başlangıçta makul tutuyor |
| BatchNorm / LayerNorm | Aynı şeyi eğitim boyunca sürekli yapıyor |
| Gradient clipping | $L$ tahmininin tuttuğu varsayımı bozulduğunda hasarı sınırlıyor |
| Learning rate warmup | Başta $L$ bilinmiyor, küçük başlayıp kademeli büyütüyorsun |
| Adam / RMSprop | Her parametre için ayrı efektif adım boyu — yön bazında farklı eğriliği telafi ediyor |

Yani $L$, kodda bir değişken olarak görünmüyor ama **learning rate'le ilgili her kararın arkasında
duran şey.** Bu bölümün amacı da o kararları tahminden çıkarıp bir gerekçeye oturtmak.

### 4.2c Düşünce zinciri — ve sınırları

Buraya kadarki akış:

```
1. lr'nin bir üst sınırı var              →  lr < 2/L
2. o sınır L'ye bağlı                     →  L büyükse lr küçük olmak zorunda
3. L verinin ölçeğine bağlı               →  BU veride λmax = 439, çünkü rezervasyonlar 2-30 arası
4. öyleyse ölçeği düzeltirsem L düşer     →  439 → 2
5. lr'yi çok büyütebilirim                →  0.00456 → 1.0  (BU veride, 220 kat)
```

**5. adımdaki `1.0` bu veriye ait bir sayı, genel bir kural değil** — sebebi hemen aşağıda.

#### Standartlaştırma ne yapıyor

Her sütunu ortalaması 0, standart sapması 1 olacak şekilde ölçeklemek:

```
Z = (X - X.mean()) / X.std()
```

| | ilk 5 değer | ortalama | std | $\lambda_{\max}$ | kritik lr |
|---|---|---|---|---|---|
| ham $X$ | `13, 2, 14, 23, 13` | 12.67 | 7.64 | 439.07 | 0.00456 |
| standartlaştırılmış $Z$ | `0.04, -1.40, 0.17, 1.35, 0.04` | 0.00 | 1.00 | 2.00 | 1.0 |

Sayılar küçüldü, **bilgi değişmedi** — hangi günün daha çok rezervasyon aldığı aynı.

#### Neyi düzeltmiyor — ve "lr=1" neden genel değil

Standartlaştırma **ölçek** farkını düzeltiyor, **özellikler arası korelasyonu** düzeltmiyor.
Bu veride `lr = 1` çıkmasının sebebi üç koşulun birden tutması: **MSE loss**, **standartlaştırılmış
özellikler**, **ve tek bir özellik olması** (dolayısıyla korelasyon sorunu yok).

Üçüncüsü gerçek veride nadiren doğru. Ölçüm (5000 örnek, hepsi standartlaştırılmış — yalnızca
korelasyon değişiyor):

| durum | $\lambda_{\max}$ | kritik lr |
|---|---|---|
| tek özellik | 2.00 | 1.0000 |
| 5 özellik, korelasyonsuz | 2.09 | 0.9568 |
| 5 özellik, korelasyon 0.9 | 9.18 | 0.2178 |
| 20 özellik, korelasyon 0.9 | 36.27 | **0.0551** |

*(Bu tablo bir ilüstrasyon, genel bir yasa değil — belirli bir sentetik kurulumda ölçüldü.
Gösterdiği tek şey: standartlaştırma tek başına condition number'ı 1'e indirmiyor.)*

Loss değişirse sınır da değişir: log loss'ta Hessian $X^\top S X$ ve
$S = \mathrm{diag}(\hat{y}(1-\hat{y})) \le 1/4$ olduğu için tavan başka çıkar.

#### Ölçek/koşullanma sorununun alternatif çözümleri

Standartlaştırma bu ailenin **bir** üyesi, tek çözüm değil:

| Yöntem | Ne yapıyor | Ne zaman |
|---|---|---|
| **Standartlaştırma** (z-score) | Ortalamayı 0, std'yi 1 yapıyor | Varsayılan tercih; ölçeği düzeltir, korelasyonu düzeltmez |
| **Min-max ölçekleme** | Değerleri `[0,1]`'e sıkıştırıyor | Sınırlı aralık gerektiğinde (görüntü pikselleri gibi) |
| **Robust ölçekleme** | Ortalama/std yerine medyan/IQR kullanıyor | Aykırı değer varsa — std'yi aykırı değer şişiriyor |
| **Whitening / PCA** | Ölçeği **ve** korelasyonu birlikte düzeltiyor, $\kappa \to 1$ | Yukarıdaki tablonun çözümü. Bedeli: $d\times d$ kovaryans matrix'i, büyük $d$'de pahalı |
| **Adaptif optimizer** (Adam, RMSprop) | Veriyi hiç değiştirmiyor; her parametre için ayrı efektif adım boyu tutuyor | Ölçeği düzeltmek yerine **etkisini telafi ediyor**. Derin öğrenmede en yaygın yol |
| **İkinci mertebe** (Newton, L-BFGS) | Eğriliği doğrudan tahmin edip adımı ona göre kuruyor | Küçük/orta problemlerde; derin ağda Hessian kurulamıyor |
| **lr taraması / lr finder** | Sınırı hesaplamıyor, **ölçüyor** | Her durumda geçerli, teorik varsayım gerektirmiyor |

**Pratikte derin öğrenmede kullanılan bileşim:** girdiyi standartlaştır + Kaiming init + BatchNorm
+ Adam + lr taraması. Hiçbiri tek başına yeterli değil; her biri sorunun farklı bir katmanına
dokunuyor.

**Ve zincir derin ağda bitmiyor, orada tekrar başlıyor:** girdiyi bir kez standartlaştırmak
yetmiyor, çünkü her layer'ın çıktısı bir sonrakinin girdisi ve ölçek layer layer kayıyor. Kaiming
init ve BatchNorm, yukarıdaki 4. adımı ağın **içinde tekrar tekrar** yapan tekniklerin adı —
[14-kaiming-init-ve-batchnorm](/posts/yz50-14-kaiming-init-ve-batchnorm/).

### 4.3 Neden böyle bir koşul **var**

Önce sınırın ne olduğunu değil, **niçin var olduğunu** soralım. Cevap şu: kuadratik bir loss'ta
gradient descent aslında bir **lineer özyineleme** ve lineer sistemlerin keskin bir kararlılık
eşiği vardır.

> **Hatırlatman gereken konular** (matematik altyapın var, sıfırdan öğrenmen gerekmiyor ama bu
> bölüm bunlara dayanıyor):
> - **Simetrik matrix'lerin spektral teoremi** — reel simetrik bir matrix ortonormal eigenvector
>   tabanında köşegenleştirilebilir
> - **Lineer özyineleme kararlılığı** — $e_{k+1} = M e_k$ ancak ve ancak $\rho(M) < 1$ ise sıfıra gider
> - **Kuadratik formlar ve pozitif tanımlılık**
> - **Condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$

**Adım 1 — özyinelemeyi kur.** Optimumdan sapmayı $e = w - w^\ast$ diye tanımla. Kuadratik loss'un
gradient'i optimumda sıfır olduğundan $\nabla L(w) = He$. Güncelleme kuralını yaz:

$$e_{k+1} = e_k - \mathrm{lr}\,\nabla L = e_k - \mathrm{lr}\,H e_k = \underbrace{(I - \mathrm{lr}\,H)}_{M}\,e_k$$

Yani hata her adımda **aynı** matrixle çarpılıyor. $k$ adım sonra $e_k = M^k e_0$. Soru artık
optimizasyon sorusu değil: $M^k$ ne zaman sıfıra gider?

**Adım 2 — yönleri birbirinden ayır.** Bu adım $H$'ın **simetrik** olmasına dayanıyor
($\partial^2 L/\partial w \partial b = \partial^2 L/\partial b \partial w$, karışık kısmi
derivative'lerin eşitliği). Spektral teorem gereği $H = Q\Lambda Q^\top$, $Q$ ortonormal. O zaman:

$$M = I - \mathrm{lr}\,H = Q\,(I - \mathrm{lr}\Lambda)\,Q^\top$$

Özvektör tabanında $M$ **köşegen**. Yani hatayı eigenvector'lere ayrıştırırsan her bileşen diğerlerinden
bağımsız olarak, her adımda tek bir sayıyla çarpılıyor:

$$e^{(i)}_{k+1} = (1 - \mathrm{lr}\,\lambda_i)\,e^{(i)}_k$$

**Koşulun var olmasının sebebi bu ayrışma.** Problem birbirine karışmış iki boyutlu bir hareket
olmaktan çıkıp, iki bağımsız geometrik diziye dönüşüyor. Geometrik dizinin ne zaman sıfıra gittiği
ise ortaokul bilgisi: çarpan mutlak değerce 1'den küçük olmalı.

**Adım 3 — hangi eigenvalue'nun bağlayıcı olduğu.** Her yönün küçülmesi gerekiyor, yani koşul
$|1 - \mathrm{lr}\lambda_i| < 1$ **her** $i$ için sağlanmalı. `lr`'yi büyüttükçe bu ilk olarak en
büyük eigenvaluede bozuluyor. Dolayısıyla bağlayıcı kısıt $\lambda_{\max}$:

$$
|1 - \mathrm{lr}\,\lambda| < 1 \quad \Longleftrightarrow \quad 0 < \mathrm{lr}\,\lambda < 2
\quad \Longrightarrow \quad \boxed{\ \mathrm{lr} < \frac{2}{\lambda_{\max}}\ }
$$

**Adım 4 — 2 sayısı nereden geliyor.** Eşitsizliği açarsan $-1 < 1 - \mathrm{lr}\lambda < 1$.
Sağ taraf `lr > 0` diyor, önemsiz. Asıl bilgi sol tarafta ve anlamı şu:

| $\mathrm{lr}\cdot\lambda$ | çarpan $1-\mathrm{lr}\lambda$ | o yönde ne oluyor |
|---|---|---|
| $\to 0$ | $\approx 1$ | neredeyse hiç ilerlemiyor |
| $= 1$ | $0$ | **tek adımda tam minimuma** düşüyor |
| $1 < \cdot < 2$ | $-1$ ile $0$ arası | minimumu aşıyor ama daha yakına düşüyor — salınarak yakınsıyor |
| $= 2$ | $-1$ | sonsuza kadar aynı iki nokta arasında zıplıyor |
| $> 2$ | $< -1$ | her aşış bir öncekinden büyük — ıraksıyor |

Yani **2, aşma bütçesi**: minimumun öte tarafına geçebilirsin, ama gittiğin mesafenin iki katından
fazla geçemezsin. `lr = 1/λ` mükemmel adım, `2/λ` kopma noktası.

**Neden bu kadar temiz:** $H$'ın parametreye bağlı olmaması (4.2) sayesinde $M$ her adımda aynı
matrix. Hessian $w$ ile değişseydi $M$ de değişir, $M^k$ diye bir şey yazamaz ve böyle kesin bir
eşik bulamazdık — ancak yerel ve yaklaşık bir şey söylenebilirdi.

Bu veride ($\overline{x^2} = 218.8$, $\bar{x} = 12.67$):

| | $\lambda_{\max}$ | $\lambda_{\min}$ | kritik lr | condition number $\kappa$ |
|---|---|---|---|---|
| ham $X$ | 439.07 | 0.5316 | **0.004555** | **825.9** |
| standartlaştırılmış $X$ | 2.00 | 2.00 | 1.0 | 1.0 |

### 4.4 İki ayrı sorun, iki ayrı eigenvalue

**Sorun 1 — patlama, $\lambda_{\max}$ yüzünden.** `lr = 0.01` kritik değerin 2.2 katı. Çarpanı
hesapla:

$$|1 - 0.01 \times 439.07| = 3.39$$

Her adımda hata dik yönde **3.39 kat büyüyor**. 15 adımda $3.39^{15} \approx 2\times10^{8}$ — koşturunca
gerçekten $\lVert w \rVert \approx 1.7\times 10^{8}$ çıkıyor. Grafikteki patlama bu sayı.

Kritik sınırın %90'ında (`lr = 0.0041`) aynı çarpan `0.80` oluyor, yani hata her adımda küçülüyor.

**Sorun 2 — yavaşlık, $\lambda_{\min}$ yüzünden.** Güvenli bir lr seçtin, ama artık **diğer**
yöndeki çarpan `0.99782`. Yani düz yönde her adım hatayı yalnızca binde iki azaltıyor; e-kat
azalma için ~459 adım gerekiyor. Tablodaki 20.000 iterasyon bundan.

İki eigenvalue birbirini kıstırıyor: lr'yi $\lambda_{\max}$ belirliyor, ilerlemeyi $\lambda_{\min}$.
Oranları **condition number** ($\kappa = 825.9$) ve gereken iterasyon sayısı kabaca $\kappa$ ile
orantılı.

### 4.5 Standartlaştırma ikisini birden çözüyor

$X$'i ortalaması 0, standart sapması 1 olacak şekilde ölçeklersen $H = 2I$ olur: iki eigenvalue de
`2.00`, $\kappa = 1$, kritik lr `1.0`.

Yani **girdinin ölçeğini düzeltmek, öğrenme hızının kendisidir** — daha iyi bir optimizer değil,
daha iyi koşullanmış bir problem.

**Aynı problem derin ağlarda da var ve iki ayrı yöntemle çözülüyor.** Orada girdiyi
standartlaştırmak yetmiyor, çünkü her layer'ın çıktısı bir sonrakinin girdisi ve ölçek layer
layer kayıyor:

- **Kaiming init** — weight'leri, aktivasyon varyansı layer'lar boyunca sabit kalacak ölçekte
  başlatmak. Yani koşullanmayı **başlangıçta** düzeltmek
- **BatchNorm** — her layer'ın çıktısını eğitim boyunca yeniden normalize etmek. Yani koşullanmayı
  **sürekli** düzeltmek

İkisi de [13-aktivasyon-fonksiyonlari](/posts/yz50-13-aktivasyon-fonksiyonlari/) ve YZ50 Hafta 4'ün konusu.
Fark şu: burada eigenvalue'ları yazıp $\kappa = 825.9$ diye görebiliyorsun; orada matrix çok büyük
olduğu için göremiyorsun, ama olan şey aynı.

*(Bu bölümün bütün sayıları veriden hesaplandı, elle yazılmadı.)*

---

## 5. Hangi loss

**Bu bölüm neden var:** MSE'yi seçmek bir karardı, varsayılan değil. Alternatifleri ve hangi durumda ne yaptıklarını bilmek gerekiyor.

MSE tek seçenek değil, sadece kapalı formu olan seçenek:

| loss | çözümü | aykırı değere tepkisi |
|---|---|---|
| MSE | ortalama | tek nokta eğimi sürükleyebilir |
| MAE (Mean Absolute Error) | medyan | her nokta eşit weightta |
| Huber | arada | $\delta$'ya kadar MSE, sonrası tavanlı |

Bu veride tek bir noktayı 75 pizza yukarı taşımak MSE'nin eğimini 1.081'den 1.695'e kaydırıyor;
MAE ve Huber neredeyse hiç oynamıyor. Detay ve ölçümler:
[17-mae-huber-loss](/posts/yz50-17-mae-huber-loss/).

---

## 6. İstatistiksel okuma

**Bu bölüm neden var:** "MSE (ortalama kare hata) neden kare alıyor" sorusunun cevabı estetik değil istatistiksel. Ayrıca modelin ne kadar iyi olduğunu tek sayıyla söylemek gerekiyor.

### Neden kare alıyoruz

**MSE** (Mean Squared Error — ortalama kare hata) keyfi bir tercih değil. Gürültü hakkında bir
varsayım yaparsan, kare almak o varsayımdan **türetiliyor**:

> Gerçek değer = modelin tahmini + gürültü, ve gürültü sıfır ortalamalı, sabit varyanslı
> **Gauss** (normal dağılım) ise → MSE'yi minimize etmek, **maximum likelihood** (en büyük
> olabilirlik) tahmini ile birebir aynı şeydir.

Bu durumda **OLS** (Ordinary Least Squares — sıradan en küçük kareler; bu notun 3. bölümündeki
kapalı form çözümünün adı) aynı zamanda istatistiğin "en iyi" dediği tahmin oluyor.

Aynı desen sigmoid+log loss ve softmax+cross-entropy için de geçerli — üçü tek bir teoremin
örnekleri (detayı [16-loss-nll-cross-entropy](/posts/yz50-16-loss-nll-cross-entropy/)'de).

### Modelin ne kadar iyi olduğunu tek sayıyla söylemek

Bu veride hesaplanan üç sayı ve okunuşları:

| Ölçü | Değer | Ne demek |
|---|---|---|
| $R^2$ (belirtme katsayısı) | **0.749** | Satılan pizza sayısındaki oynamanın **%75'i** rezervasyon sayısıyla açıklanıyor. Kalan %25 başka şeylerden (hava, gün, tesadüf) |
| Artık standart sapması | **4.95 pizza** | Model tipik olarak **±5 pizza** yanılıyor. "20 rezervasyon → 35 pizza" dediğinde gerçek değer kabaca 30-40 arası |
| Korelasyon $r$ | **0.866** | İki sütun arasındaki doğrusal ilişkinin gücü; 1 mükemmel, 0 ilişkisiz |

**Somut okunuşu:** elinde hiçbir model yokken en iyi tahminin "ortalama kadar pizza" olurdu ve
tipik hatan $\approx$ 9.9 pizza olurdu. Rezervasyon sayısını kullanan bu model hatayı **4.95'e**
indiriyor — yarı yarıya.

**İki teknik ayrıntı:**

- Artık standart sapmasının paydasında $n$ değil **$n-2$** var. Sebep: iki parametre ($w$ ve $b$)
  aynı veriden kestirildi, yani veri "iki kez kullanıldı" ve serbestlik derecesi o kadar düştü.
  $n$ ile bölersen hatayı sistematik olarak olduğundan küçük gösterirsin
- $R^2 = r^2$ — bu tek girdili regresyonda **özdeşlik**, tesadüf değil: $0.866^2 = 0.749$. Çok
  girdili halde bozulur

---

## 7. Bu zaten bir nöron

**Bu bölüm neden var:** Buraya kadarki her şeyin sinir ağlarında aynen geçerli olduğunu, ve tam olarak neyin değiştiğini tek yerde toplamak için.

$\hat{y} = wx + b$, aktivasyon fonksiyonu olmayan tek bir nörondur. Üstüne `tanh` koyarsan bir
nöron, yan yana dizersen bir layer, layer'ları üst üste koyarsan MLP (Multi-Layer Perceptron) olur.

Bu yüzden lineer regresyonda öğrenilen her şey aynen taşınıyor: parametre/loss ayrımı, gradient'in
yönü, learning rate'in kararlılık sınırı, girdinin ölçeğinin önemi, loss seçiminin aykırı
değerlerle ilişkisi. Değişen tek şey, kapalı form çözümün ortadan kalkması.

---

## Özet

1. Lineerlik girdide değil parametrede; `tanh` eklemek çizgiyi geçiyor.
2. Koordinat araması çalışır ama parametre sayısıyla ölçeklenmez — gradient'in varlık sebebi bu.
3. Kapalı form $\mathrm{Cov}/\mathrm{Var}$; toplam ve ortalama formunu karıştırmak sessiz hata.
   Kapalı formu bitiren şey **süre değil bellek** ($d\times d$ matrix) ve koşullanmanın kareye
   çıkması — ölçüldü.
4. Hessian **parametrelere göre** sabit, veriye göre değil — bu yüzden lr'nin üst sınırı
   $2/\lambda_{\max}$ eğitim başlamadan hesaplanabiliyor ve her yerde geçerli.
5. $\lambda_{\max}$ patlamayı, $\lambda_{\min}$ yavaşlığı belirliyor; standartlaştırma koşul
   sayısını 826'dan 1'e indirip ikisini birden çözüyor — normalizasyon fikrinin çekirdeği.
6. MSE, Gauss gürültü varsayımı altında MLE; loss seçimi bir dağılım varsayımı.

## İlgili notlar

- [04-sinir-agi-nedir](/posts/yz50-04-sinir-agi-nedir/) — aynı yapının aktivasyonlu hali
- [06-gradient-descent](/posts/yz50-06-gradient-descent/) — gradient sezgisi ve numerical derivative'in sınırları
- [07-neden-gradient-descent](/posts/yz50-07-neden-gradient-descent/) — kapalı form neden genel çözüm değil
- [15-aktivasyon-loss-eslesmesi](/posts/yz50-15-aktivasyon-loss-eslesmesi/) — MSE'nin GLM'deki yeri
- [17-mae-huber-loss](/posts/yz50-17-mae-huber-loss/) — MSE dışındaki loss'lar ve aykırı değerler
