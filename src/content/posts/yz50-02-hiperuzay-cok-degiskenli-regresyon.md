---
title: "Hiperuzay — Çok Değişkenli Regresyon ve Matrix Gradient'i"
published: 2026-09-29
description: "Tek girdiden çok girdiye geçiş: tasarım matrisi, bias trick, matrix formunda gradient ve kapalı form çözümün ne zaman iteratif yönteme yenildiği."
tags:
  - YZ50
  - Lineer Regresyon
  - Matris
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

Tek girdiden çok girdiye geçiş. Görünüşte küçük bir genelleme, pratikte üç şeyi birden
değiştiriyor: kodun döngüden matrix çarpımına inmesi, gradient'in transpozla yazılması, ve özellik
ölçeklerinin artık isteğe bağlı olmaktan çıkması.

Devamı olduğu not: [01-lineer-regresyon](/posts/yz50-01-lineer-regresyon/).

---

## 1. Doğru değil, hiperdüzlem

**Bu bölüm neden var:** Tek girdiden çok girdiye geçmek geometrik olarak ne demek, ve bu geçiş algoritmayı mı yoksa sadece senin görselleştirme yeteneğini mi zorluyor?

Girdi tek sütun olmaktan çıkıp birkaç sütuna dönüşünce:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

2 boyutta bir doğru, 3 boyutta bir düzlem, $n$ boyutta çizemediğin bir hiperdüzlem. İsim buradan.

![Gradient descent bir doğruyu değil düzlemi oturtuyor](/yz50/hiperuzay-duzlem-fit.gif)

Asıl mesaj şu: **boyut sayısı senin için problem, algoritma için değil.** Gradient descent'in
kodu neredeyse hiç değişmiyor — aynı döngü, aynı güncelleme kuralı. Değişen tek şey, artık
göremiyor olman.

*(GIF'teki ikinci değişken — sıcaklık — sentetiktir; pizza verisinde böyle bir sütun yok,
geometriyi göstermek için sabit tohumla üretildi.)*

---

## 2. Bias hilesi: 1'ler sütunu

**Bu bölüm neden var:** İki parametreyi (`w` ve `b`) ayrı ayrı taşımak, çok değişkenliye geçince iki ayrı gradient kolu yazmak demek. Bu bölüm o ikiliği tek bir matrix çarpımına indiren numarayı ve **bedelini** gösteriyor — çünkü sinir ağlarında bu numaradan vazgeçiliyor.

$b$'yi ayrı bir değişken olarak taşımak yerine $X$'e sabit 1'lerden oluşan bir sütun eklenir:

$$\hat{y} = \sum_{j=0}^{n} w_j x_j, \qquad x_0 \equiv 1$$

Böylece $w_0$ otomatik olarak bias oluyor. Kazanç: iki ayrı işlem (`X @ w + b`) yerine tek işlem
(`X @ w`) kalıyor, gradient da tek formüle iniyor — bias için ayrı bir kol yazmıyorsun.

**Sinir ağlarında bu hileden vazgeçiliyor**, `b` ayrı duruyor. Sebep: çok layer'lı yapıda her
layer'ın çıktısına 1'ler sütunu eklemek, broadcasting ile yapılan bias toplamadan hem pahalı hem
çirkin. Tek layer'da şık, derinlikte değil. `embcat @ W1 + b1` yazarken bunu görüyorsun.

### Sık karışan nokta: 1'ler sütunu bias değildir *(2026-09-18)*

Kodda `loss(X, Y, w)` imzasında `b` görünmeyince "bias'ı optimize etmiyoruz, sabit tutuyoruz"
hissi doğuyor. Yanlış his. Karışan şey iki farklı nesnenin aynı sanılması:

| | Nedir | Değişir mi |
|---|---|---|
| `X`'in ilk sütunu (hep 1) | **veri** — değeri her zaman 1 olan sahte bir özellik | ❌ sabit |
| `w[0]` | **parametre** — eski `b` | ✅ her iterasyonda güncellenir |

Sabit olan 1'ler sütunu; `w[0]` değil.

**Neden bias'ı bir weight gibi yazabiliyoruz.** `b` yerine `1 * b` yaz — değeri değişmez, şekli
değişir:

```
tahmin = x * w + b      →      tahmin = x * w + 1 * b
```

Artık ikisi de aynı kalıpta: *girdi × weight*. `x`'in weight'i `w`, `1`'in weight'i `b`. O yüzden
girdileri `[1, x1, x2, x3]`, weight'leri `[b, w1, w2, w3]` diye tek listeye koyup tek çarpımda
halledebiliyorsun. `b` kaybolmadı, **adı `w[0]` oldu.**

**Loss `b`'ye bağlı, sadece görünmüyor.** `loss(X, Y, w)` → `predict(X, w)` → `X @ w`, ve bu çarpım
her satır için şunu açıyor:

```
1*w[0] + x1*w[1] + x2*w[2] + x3*w[3]
```

Baştaki `1*w[0]` bias. Kontrolü ucuz: eğitim bitince `w[0]`'a elle 100 ekle, `loss`'u tekrar
yazdır — fırlar. Bağlı olmasa kıpırdamazdı.

**Ayrı yazılan `b` derivative'i nereye gitti.** Tek değişkenli kodda bias'ın gradient'i ayrı bir satırdı:

```
b_gradient = 2 * ortalama(tahmin - Y)      # hataların ortalaması
```

Matrix halinde öyle bir satır yok, tek bir `X.T @ (tahmin - Y)` var. Ama o çarpımın **ilk satırına**
bak: `X.T`'nin ilk satırı = `X`'in ilk sütunu = hepsi 1. Yani o satır şunu hesaplıyor:

```
1*hata1 + 1*hata2 + ... = hataların toplamı
```

Örnek sayısına bölüp 2 ile çarpınca **tam olarak yukarıdaki `b_gradient`**. Diğer satırlarda 1
yerine `x1`, `x2`, `x3` olduğu için oralarda `w`'lerin gradient'i çıkıyor. Yeni bir formül
türetilmedi; aynı hesap matrix çarpımının içine gömüldü.

Matematiksel yazımı (bölüm 4'teki gösterimle):

$$\left(X^\top(\hat{y}-y)\right)_0 = \sum_i 1\cdot(\hat{y}_i - y_i) = \sum_i \text{hata}_i$$

**Özetle üç cümle:** `b` silinmedi, `w[0]` oldu. Sabit tutulmuyor, diğer weight'lerle aynı satırda
(`w -= gradient * lr`) güncelleniyor. Loss'ta var, `matmul`'ün içinde.

---

## 3. Şekiller

**Bu bölüm neden var:** Matrix boyutlarını takip etmek bu noktada sıkıcı bir muhasebe gibi görünüyor. Bölüm 4'te bunun aslında **derivative bulma aracı** olduğu ortaya çıkacak — şekil disiplini oraya hazırlık.

$$\underbrace{X}_{(m,\,n)} \cdot \underbrace{w}_{(n,\,1)} = \underbrace{\hat{y}}_{(m,\,1)}$$

$m$ örnek sayısı, $n$ özellik sayısı. `predict` tek satır: `X @ w`. Döngü yok — 30 örnek de olsa
30 milyon da olsa aynı ifade. Bu vektörleşme olmadan MNIST'e (784 özellik) geçilemiyor.

---

## 4. Gradient'in matrix hali

**Bu bölüm neden var:** Tek değişkenli gradient formülünü ezberlemiştin. Matrix haline geçerken transpose nereden çıkıyor ve nereye konuyor? Bu soru elle backpropagation yazarken her satırda karşına çıkacak, o yüzden **ezber yerine türetme yöntemi** kuruyoruz.

$$L = \frac{1}{m}\lVert Xw - y \rVert^2 \quad\Longrightarrow\quad \nabla_w L = \frac{2}{m} X^\top (Xw - y)$$

Tek değişkenli halde gradient $\frac{2}{m}\sum x_i(\hat{y}_i - y_i)$ idi — "hatayı girdiyle çarp,
topla". Matrix halinde toplama işini **transpoz** yapıyor. $X^\top$ burada tesadüf değil, genel
bir kuralın örneği.

### Genel kural ve şekilden türetme

$C = A B$ ise:

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^\top, \qquad
\frac{\partial L}{\partial B} = A^\top \frac{\partial L}{\partial C}
$$

Ezberlemeye gerek yok, şekiller tek bir probability bırakıyor:

![dA = dC @ B^T kuralı şekillerden çıkıyor](/yz50/matrix-gradient-sekil.gif)

**Neden böyle bir kural var — ve neden şekil bakmak yetiyor:**

> **Hatırlatman gereken konular:** matrix çarpımında boyut uyumu ($(m,k)\cdot(k,n) = (m,n)$),
> transpose'un boyutları ters çevirmesi, ve bir fonksiyonun derivative'inin argümanıyla aynı uzayda
> yaşaması.

Kural bir tesadüf değil, iki gerçekten çıkıyor:

1. **Gradient, kendi tensörüyle aynı şekildedir.** Çünkü gradient "bu tensörün her bir elemanını
   oynatırsam loss ne kadar değişir" sorusunun cevabı — eleman sayısı kadar cevap var, dolayısıyla
   şekil de aynı olmak zorunda.
2. **Elinde yalnızca iki malzeme var:** yukarıdan gelen $dC$ ve çarpımın diğer tarafı. Bunları
   çarpmanın sınırlı sayıda yolu var ve çoğu boyut uyumsuzluğundan tanımsız.

İkisi birleşince geriye tek bir geçerli kombinasyon kalıyor. Yani şekil kontrolü derivative'i
*türetmiyor*, **elemeyi yapıyor** — ve tek aday kaldığı için doğru olanı buluyor.

Akıl yürütme: **bir gradient her zaman kendi tensörüyle aynı şekildedir.** $A$ $(4,3)$ ise $dA$ da
$(4,3)$ olmak zorunda. Elinde $dC$ $(4,2)$ ve $B$ $(3,2)$ var. $dC \cdot B$ tanımsız (iç boyutlar
2 ve 3). $dC \cdot B^\top$ ise $(4,2)\cdot(2,3) = (4,3)$ — tutan tek kombinasyon. Transpozun
nereye geleceğini ve çarpımın hangi taraftan olacağını şekil belirliyor.

Bu, elle backprop yazarken doğrudan kullanılan yöntem: derivative'i tam türetemediğinde şekilleri yaz,
tutan tek kombinasyonu seç, sonra sayısal olarak doğrula. `dW1 = embcat.T @ dhprebn` satırının
gerekçesi bu.

İki uyarı:

- Şekil tutması **gerekli** koşul, yeterli değil. $(4,3)$ üreten yanlış bir ifade de yazabilirsin
  (örneğin bir yerde `sum` unutmak). Şekil hatayı ucuza eler, doğruluğu kanıtlamaz — sayısal
  karşılaştırma yine şart.
- Broadcasting varsa geri dönüşte o eksende `sum` gerekir: forward pass'te $(1,n)$'lik bir bias
  $(m,n)$'e kopyalandıysa, gradient geri dönerken $m$ ekseni boyunca toplanır. Yoksa şekil
  tutmaz — ki bu da şekil disiplininin yakaladığı hatalardan.

---

## 5. Özellik ölçekleri artık pazarlık konusu değil

**Bu bölüm neden var:** Tek değişkenlide standartlaştırma bir iyileştirmeydi, atlanabilirdi. Çok değişkenlide neden **zorunlu** hale geldiğini göstermezsek, ileride Kaiming init ve BatchNorm'un varlık sebebi havada kalır.

[01](/posts/yz50-01-lineer-regresyon/)'de tek değişkenli halde standartlaştırma bir iyileştirmeydi. Çok
değişkenlide **zorunlu**, çünkü Hessian $\frac{2}{m}X^\top X$ ve en büyük ölçekli sütun
$\lambda_{\max}$'ı tek başına belirliyor. Rezervasyon ~10, sıcaklık ~20, turist ~1000
mertebesindeyse learning rate'i en büyük ölçekli özelliğe göre kısmak zorundasın — diğerleri o
küçük adımlarla neredeyse hiç öğrenmez.

![Ölçekler farklıyken loss vadisi basık, GD zigzaglıyor](/yz50/ozellik-olcegi-zigzag.gif)

İki panel aynı problem, aynı veri, **aynı başlangıç loss'u**; her biri kendi kritik lr'sinin
%90'ıyla koşuyor. Tek fark özelliklerin ölçeği:

| | condition number $\kappa$ | lr | 14. adımda optimumun üstündeki loss |
|---|---|---|---|
| ham özellikler | 58 | 0.0145 | 63.96 |
| standartlaştırılmış | 1 | 0.8692 | 0.59 |

Solda olan şey klasik: dik yönde her adım karşı yamaca sekiyor (zigzag), düz yönde ise
sürünüyor — ilerleme oradan gelmesi gerekirken. Sağda konturlar çembere dönüşünce gradient
doğrudan merkezi gösteriyor, yol düz.

Bu fikrin derin ağlardaki hali Kaiming init ve BatchNorm. Burada iki parametreyle çizilebiliyor;
orada çizilemiyor ama aynı şey oluyor.

---

## 6. İki yeni tuzak

**Bu bölüm neden var:** Boyut artışı yalnızca ölçek sorunu getirmiyor, iki yeni başarısızlık modu da açıyor. İkisi de sessiz — hata vermiyorlar.

**Eşdoğrusallık (collinearity).** İki özellik birbiriyle yüksek korelasyonluysa $X^\top X$
tekile yaklaşıyor: kapalı form ya çöküyor ya da birbirini götüren devasa katsayılar üretiyor
(biri +1000, diğeri −998 gibi). Gradient descent patlamıyor, sadece yavaşlıyor — çünkü küçük
eigenvalue sıfıra yaklaşırken $\kappa$ büyüyor. İteratif yöntemin kapalı forma karşı ikinci
üstünlüğü bu; birincisi $\mathcal{O}(d^3)$ maliyetti.

**Özellik sayısı arttıkça overfit.** Tek değişkende zor olan şey, 50 değişkende kolay: model
gürültüyü ezberleyebiliyor. Train/dev ayrımının gerekçesi burada başlıyor.

---

## 7. Nerede karşına çıkıyor

**Bu bölüm neden var:** Buradaki soyut matrix cebrinin, YZ50'de yazdığın kodun hangi satırına denk geldiğini göstermek için.

makemore'daki

```
h = torch.tanh(embcat @ W1 + b1)
```

satırındaki `embcat @ W1 + b1` tam olarak bu notun konusu: 30 özellik (3 harf × 10 boyutlu
embedding), 200 çıktı. Üstüne `tanh` gelince lineer regresyon olmaktan çıkıyor
([01](/posts/yz50-01-lineer-regresyon/), bölüm 1), ama altındaki matrix mekaniği aynen duruyor. Yani
buradaki şekil disiplini, oradaki forward pass'in okunabilmesi demek.

## Özet

1. Çok değişkenli regresyon bir hiperdüzlem oturtuyor; algoritma boyuttan etkilenmiyor.
2. 1'ler sütunu bias'ı weight'lerin içine katıyor — tek layer'da şık, derin ağda terk ediliyor.
   1'ler sütunu **veri**, $w_0$ **parametre**; $b$ kaybolmadı, $w_0$ oldu ve her adımda güncelleniyor.
   $b$'nin derivative'i $X^\top$'un ilk satırından kendiliğinden düşüyor.
3. $\nabla_w L = \frac{2}{m}X^\top(Xw - y)$; transpoz, toplama işini yapan şey.
4. $dA = dC\,B^\top$ ve $dB = A^\top dC$ — şekillerden çıkıyor, ezber gerekmiyor; şekil tutması
   gerekli ama yeterli değil.
5. Ölçek farkı condition number'ı şişiriyor, GD zigzaglıyor: 58 → 1 ile lr 60 kat artıyor.
6. Eşdoğrusallık kapalı formu kırıyor, GD'yi sadece yavaşlatıyor.

## İlgili notlar

- [01-lineer-regresyon](/posts/yz50-01-lineer-regresyon/) — tek değişkenli hali, normal denklemler,
  koşullanmanın ilk görünüşü
- [12-backpropagation-calculus](/posts/yz50-12-backpropagation-calculus/) — chain rule'un skaler hali;
  buradaki matrix kuralı onun tensör karşılığı
- [18-broadcasting](/posts/yz50-18-broadcasting/) — geri dönüşte `sum`'ın nereden çıktığı
- [17-mae-huber-loss](/posts/yz50-17-mae-huber-loss/) — loss seçimi bu genellemeden bağımsız

Görseller [img/make_hyperspace_figures.py](/yz50/make_hyperspace_figures.py) ile üretiliyor.
