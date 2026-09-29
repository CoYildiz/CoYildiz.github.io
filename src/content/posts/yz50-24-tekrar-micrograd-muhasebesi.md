---
title: "Genel Tekrar — Hafta 1 ve Hafta 2"
published: 2026-09-29
description: "Hafta 1 ve 2'nin toplu tekrarı: micrograd'ın defter tutma mantığı üzerinden forward ve backward'ın uçtan uca izi."
tags:
  - YZ50
  - Backpropagation
  - Tekrar
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

Bu not bir kavram anlatımı değil, **kendi yazdığın kodun muhasebesi**: ne kurdun, neden kurdun, hangi aracı neden seçtin, ve her parçanın bir sonrakini nasıl mümkün kıldığı.

Kod: [[`hafta_1/hafta1.py`](https://github.com/CoYildiz/yz50/blob/main/hafta_1/hafta1.py)](https://github.com/CoYildiz/yz50/blob/main/hafta_1/hafta1.py) ve [[`hafta_2_backpropagation/micrograd_from_scratch.py`](https://github.com/CoYildiz/yz50/blob/main/hafta_2_backpropagation/micrograd_from_scratch.py)](https://github.com/CoYildiz/yz50/blob/main/hafta_2_backpropagation/micrograd_from_scratch.py)

---

## Hafta 1 — "Bir ağ nedir, öğrenmek ne demek"

**Amaç:** Kütüphane kullanmadan, bir nöronun ne yaptığını ve "öğrenme" denen şeyin somut olarak ne olduğunu görmek.

### Ne kurdun

**1. Tek nöronun ileri geçişi** — `forward_pass_basic()`

Girdileri weight'lerle çarpıp topladın, bias ekledin, sigmoid'den geçirdin. Tek satırlık fikir: bir nöron, **ağırlıklı toplam + sıkıştırma**.

```python
sumofiw = 0
for i, w in zip(inputs, weights):
    sumofiw += i * w
sumofiw += bias
return sigmoid(sumofiw)
```

`zip` kullanman tesadüf değil — girdi ve weight **eşleşen** iki liste, ikisini aynı anda gezmek gerekiyor.

**2. Layer** — `forward_pass()`

Aynı işlemi bir weight listesi üzerinde tekrarladın. Layer, ayrı bir kavram değil: **aynı nöronun farklı weight'lerle çoğaltılmış hali**. Beş nöronluk bir layer'ın çıktısı, beş elemanlı bir liste.

**3. Loss** — `loss_func()`

Karelerin toplamı. Kare almanın iki sebebi var: işareti yok ediyor (fazla tahmin ile eksik tahmin eşit cezalandırılıyor) ve derivative'i sürekli (mutlak değerin köşesi yok).

**4. Loss eğrisi** — `matplotlib_draw()`

Tek bir weight'i `[-2, -1, 0, 1, 2]` aralığında gezdirip loss'u çizdirdin. Bu görselleştirmenin amacı şuydu: **loss, weight'lerin bir fonksiyonu.** Soyut bir "hata" değil, üzerinde yürünebilen bir yüzey.

**5. Numerical derivativele gradient descent** — `optimize_weight()`

Burada asıl fikir oturuyor:

```python
loss_simdi  = loss_func(...)
weights[2][0] += h
loss_nudge  = loss_func(...)
weights[2][0] -= h

turev = (loss_nudge - loss_simdi) / h
weights[2][0] -= learning_rate * turev
```

Ağırlığı küçük bir `h` kadar oynat, loss'un ne kadar değiştiğine bak, farkı `h`'ye böl. Bu, derivative'in tanımının ta kendisi — limit almadan, sayısal olarak.

### Neden numerical derivative

Çünkü Hafta 1'in amacı **gradient'in ne olduğunu** hissettirmekti: "bu weight'e dokunursam loss ne kadar değişir". Numerical derivative bunu en çıplak haliyle gösteriyor, hiçbir teori gerektirmiyor.

### Neden bu yöntem yetmiyor

İki sorunu var, ikisi de Hafta 2'nin gerekçesi:

- **Maliyet.** Her weight için ağı bir kez daha çalıştırman gerekiyor. 13.002 parametreli bir ağda her adımda 13.002 forward pass demek — imkânsız.
- **Hassasiyet.** `h` çok büyükse yaklaşım bozuk, çok küçükse kayan nokta hassasiyeti kayboluyor. İki hata arasında sıkışıyorsun.

![Numerical derivativein h ikilemi](/yz50/numerical-derivative-h-ikilemi.png)

İkilem ölçüldü: `f(x) = x³`, `x = 2`, gerçek derivative `12`. `h` sağdan sola küçülürken hata önce
düşüyor (kesme hatası azalıyor), sonra **geri tırmanıyor** — çünkü `f(x+h)` ile `f(x)` birbirine
o kadar yaklaşıyor ki farkları kayan noktada anlamlı basamaklarını kaybediyor.

En iyi `h` yaklaşık `2×10⁻⁹`, ve orada bile hata sıfır değil. Merkezi fark (turuncu) daha iyi ama
iki kat maliyetli ve aynı duvara çarpıyor. **Hiçbir `h` seçimi bu duvarı kaldırmıyor** — Hafta
2'nin analytic derivativei bu yüzden gerekli.

---

## Hafta 2 — "Derivative'i ağın kendisi hesaplasın"

**Amaç:** Numerical derivativein yerine **analitik** derivative koymak, ve bunu her işlem için elle değil, otomatik yapan bir mekanizma kurmak.

### Ne kurdun

**1. `Value` sınıfı — sayı değil, sayı + geçmişi**

Normal bir `float` sadece değerini bilir. Senin `Value`'n dört şey taşıyor:

```python
self.data      # degerin kendisi
self.grad      # bu degerin cikti uzerindeki etkisi
self._prev     # bu degeri ureten Value'lar
self._op       # hangi islemden ciktigi
```

`_prev` ve `_op` sayesinde her sayı, **nereden geldiğini** hatırlıyor. Bütün hesap bittiğinde elinde bir sayı değil, bir **graf** oluyor — computation graph.

**2. Her işlem, kendi derivative'ini yanına yazıyor**

Toplama örneği:

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad  += out.grad * 1.0
        other.grad += out.grad * 1.0
    out._backward = _backward
    return out
```

Üç ayrı şey oluyor burada:

- Sonuç `Value`'su üretiliyor, girdilerini `(self, other)` olarak saklıyor
- O sonuca **özel** bir `_backward` fonksiyonu tanımlanıyor — toplamanın yerel derivative'i 1, o yüzden gelen gradient iki girdiye de olduğu gibi aktarılıyor
- Bu fonksiyon `out`'un içine iliştiriliyor, hemen çağrılmıyor

Çarpmada yerel derivative farklı: `self.grad += out.grad * other.data`. Bir çarpanın etkisi, **diğer çarpanın büyüklüğü kadar**.

**3. `+=` — üzerine yazma, topla**

Her `_backward` içinde `=` değil `+=` var. Sebebi: bir değişken hesapta birden fazla yerde kullanılıyorsa, çıktıya **birden fazla yoldan** etki ediyor demektir. Her yolun katkısı ayrı ayrı gelip toplanmalı. `=` yazsaydın son gelen yol diğerlerini silerdi — sessiz ve bulunması zor bir hata.

![Geri yayılımın muhasebesi, adım adım](/yz50/backpropagation-muhasebesi.gif)

`a` hem `d = a·b` hem `e = a+b` içinde geçiyor, yani `L`'ye **iki koldan** etki ediyor. Animasyon
gradient'lerin ters topolojik sırada nasıl dolduğunu izliyor; kritik kare beşincisi: `a.grad` önce
`d` kolundan `-2` alıyor, sonra `e` kolundan `+1` **ekleniyor** ve `-1` oluyor.

`=` yazsaydın ikinci kol birinciyi silecek ve `a.grad` `1` kalacaktı — doğru cevap `-1`. Kod
çalışmaya devam eder, sadece yanlış öğrenir.

Aynı görsel `backward()`'ın sıralama zorunluluğunu da gösteriyor: `d`'nin gradient'i hesaplanmadan
`a`'nınki hesaplanamaz, çünkü `a` kendi payını `d.grad`'dan alıyor.

**4. `backward()` — sıralama problemi**

```python
def backward(self):
    topo = []; visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

Bir düğümün gradient'inı hesaplayabilmek için **kendisinden sonraki** düğümlerin gradient'leri hazır olmalı. Topolojik sıralama tam olarak bunu garanti ediyor: her düğüm, kendisine bağlı olan herkesten sonra geliyor. Ters çevirince doğru sıra çıkıyor.

`self.grad = 1.0` satırı başlangıç koşulu: çıktının kendi kendine göre derivative'i 1.

**5. Primitifler ve türetilenler**

Sınıfta gerçekten "temel" olan az sayıda işlem var: `__add__`, `__mul__`, `__pow__`, `exp`, `tanh`. Geri kalanı bunlardan türetiliyor:

```python
def __neg__(self):      return self * -1
def __sub__(self, o):   return self + (-o)
def __truediv__(self, o): return self * o**-1
```

Çıkarma ayrı bir işlem değil, negatifle toplama. Bölme ayrı bir işlem değil, `-1` kuvvetiyle çarpma. **Her türetilen işlem, gradient'inı bedavaya alıyor** — zaten derivative'i tanımlı primitiflerden kurulduğu için.

Bu, Görev 4'ün `tanh`'ı parçalama fikrinin de özü: `tanh`'ı tek bir primitif olarak da yazabilirsin, `exp`/bölme/çıkarma ile de kurabilirsin. İkisi aynı gradient'i vermeli — verdiğini de gösterdin.

**6. `__rmul__`, `__radd__`, `__rtruediv__` — neden gerekli**

`2 * value` yazdığında Python önce `int.__mul__(2, value)` deniyor, o `NotImplemented` dönüyor, sonra `Value.__rmul__` devreye giriyor. Bu metotlar olmadan `Value` sadece kendi türüyle işlem yapabilirdi.

**7. `Neuron` / `Layer` / `MLP (Multi-Layer Perceptron)`**

Hafta 1'de nöron bir **fonksiyondu**. Burada **nesne** oldu — çünkü artık kendi weight'lerinı taşıması ve o weight'lerin eğitim boyunca **hayatta kalması** gerekiyor.

```python
class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Value(np.random.uniform(-1,1))
```

`parameters()` zinciri (`MLP` → `Layer` → `Neuron`) bütün weight'leri tek bir listede topluyor. Eğitim döngüsünün hepsine aynı anda dokunması gerektiği için şart.

**8. Eğitim döngüsü**

```python
for k in range(30):
    ypred = [c(x) for x in xs]
    loss = sum([(yp - y)**2 for y, yp in zip(ys, ypred)])
    for p in c.parameters(): p.grad = 0.0
    loss.backward()
    for p in c.parameters(): p.data += -0.07 * p.grad
```

Dört adım, sırası önemli: **forward pass → gradient'leri sıfırla → backward pass → güncelle.**

Sıfırlama neden ileri geçişten sonra: `backward()` gradient'leri `+=` ile biriktiriyor. Sıfırlamazsan önceki adımın gradient'leri bu adımınkilerle toplanır, model giderek daha yanlış yönde adım atar. Videodaki meşhur bug bu.

Sonuç: loss 2.1125 → 0.0169, tahminler `[0.94, -0.95, -0.92, 0.93]`, hedef `[1, -1, -1, 1]`.

---

## İki haftanın bağlantısı

**Bu bölüm neden var:** İki hafta ayrı konular gibi görünüyor ama biri diğerinin doğrudan cevabı: Hafta 1 bir yöntemin sınırını gösterdi, Hafta 2 o sınırı aşan yöntemi kurdu.

| | Hafta 1 | Hafta 2 |
|---|---|---|
| Derivative | Sayısal — ağı iki kez çalıştır, farkı al | Analitik — her işlem kendi derivative'ini biliyor |
| Maliyet | Parametre sayısı kadar forward pass | Tek ileri + tek backward pass |
| Nöron | Fonksiyon, weight'ler dışarıda | Nesne, weight'ler içinde ve kalıcı |
| Kapsam | Tek weight'i elle oynatmak | Bütün ağı aynı anda eğitmek |

Hafta 1 **neyi** hesaplamak istediğini gösterdi, Hafta 2 onu **nasıl ucuza** hesaplayacağını.

---

## Yakalanan hatalar ve öğrettikleri

**Bu bölüm neden var:** Çalışan koddan çok, **kırılan** koddan öğreniliyor. Bu bölüm o hataları kalıcı hale getiriyor — aynı hatayı tensör düzeyinde tekrar yapmamak için.

| Hata | Belirtisi | Ders |
|---|---|---|
| `zero_grad` çağrılmaması | Loss düşmüyor veya zıplıyor | Gradientlar birikir, her adımda sıfırlanmalı |
| `zero_grad`'ın `_backward`'ı da silmesi | İkinci `backward()` hiçbir şey yapmıyor | Sıfırlanacak olan `grad`, düğümün kendisi değil |
| Ters yazılmış `__rtruediv__` | Sayısal olarak yanlış ama sessiz sonuç | Ters operatörler ayrı düşünülmeli |
| `n.backward()` fazladan çağrısı | `Trying to backward through the graph a second time` | PyTorch geri geçişten sonra grafı serbest bırakıyor |
| `settings.database.url` | `AttributeError` | Zincirin **ilk** kopan halkasına bak |

Bu hataların ortak yanı: **üçü sessiz, ikisi gürültülü.** Sessiz olanlar çalışan ama yanlış sonuç üreten türden — Görev 4'ün üçlü doğrulaması tam olarak bunları yakalamak için var.

---

## Hafta 3'e ne taşındı

**Bu bölüm neden var:** Micrograd'da skalerle yapılan her şey, bir sonraki haftada tensörle tekrarlanıyor. Neyin aynı kaldığını bilmek, yeni kodu sıfırdan öğrenmek yerine tanımayı sağlıyor.

- `backward()`'ın mekanizması → PyTorch'un `autograd`'ı aynı fikrin tensör ölçeğindeki hali
- Eğitim döngüsünün dört adımı → aynen geçerli, sadece `p.grad = 0.0` yerine `W.grad = None`
- Gradient toplama (`+=`) → tensörlerde de aynı, `zero_grad` orada da şart
- Loss'un bir sayı olması → NLL (Negative Log Likelihood) de tek bir sayı, sadece farklı bir formülle

**Bağlantılı:** [Hafta 2 — Backpropagation Calculus (3Blue1Brown, kesin matematik)](/posts/yz50-12-backpropagation-calculus/) · [Hafta 2 — Aktivasyon Fonksiyonları: Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/) · [Hafta 3 — PyTorch Broadcasting ve `keepdim` Tuzağı](/posts/yz50-18-broadcasting/)

Görseller [img/make_micrograd_figures.py](/yz50/make_micrograd_figures.py) ile üretiliyor.
