---
title: "14 · Kaiming Init ve BatchNorm"
published: 2026-09-29
description: "Ölçek problemi ağın içinde nasıl kontrol ediliyor: Kaiming/Xavier initialization'ın türetimi ve BatchNorm'un tam olarak neyi sabitlediği."
tags:
  - YZ50
  - Normalizasyon
  - Initialization
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

[01 · Lineer Regresyon](/posts/yz50-01-lineer-regresyon/) notunun 4. bölümünde şu zincir kurulmuştu: learning
rate'in üst sınırı $2/L$, $L$ verinin ölçeğine bağlı, öyleyse veriyi standartlaştırırsan çok daha
büyük adım atabilirsin (o veride 220 kat).

**Bu not o zincirin derin ağdaki devamı.** Çünkü girdiyi bir kez standartlaştırmak yetmiyor:
her layer'ın çıktısı bir sonrakinin girdisi ve ölçek layer layer kayıyor. Kaiming init ve BatchNorm,
aynı işi ağın **içinde** yapan iki tekniğin adı.

Kaynak: Karpathy, *Building makemore Part 3*. Kendi kodun: [`hafta_4_p2-mlp/mlp-2.py`](https://github.com/CoYildiz/yz50/blob/main/hafta_4_p2-mlp/mlp-2.py).

---

## 1. Problem: ölçek layer layer kayıyor

**Bu bölüm neden var:** Kaiming ve BatchNorm birer çözüm. Önce neyin çözüldüğünü görmek gerekiyor,
yoksa formüller havada kalıyor.

Bir layer'ın çıktısı: `h = tanh(x @ W + b)`. Burada iki şey birbirini kovalıyor:

- `x` büyükse `z = x @ W + b` büyük olur
- `z` büyükse `tanh(z)` doyar → derivative sıfıra gider → **vanishing gradient**
  (bkz. [13 · Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/))

Ve tersi de kötü: `W` çok küçükse `z` sıfıra çöker, aktivasyonlar birbirine benzer, layer hiçbir
şey ayırt etmez.

**Asıl sinsi kısım derinlikte.** Tek layer'da ölçeği elle ayarlayabilirsin. Ama 6 layer varsa her
layer bir öncekinin çıktısını alıyor: ölçek her adımda biraz büyüyor ya da biraz küçülüyorsa,
6 layer sonra **üstel olarak** patlamış ya da sönmüş oluyor.

Bu yüzden soru "iyi bir başlangıç ölçeği nedir" değil, **"ölçeği layer'lar boyunca sabit tutan
başlangıç nedir"**.

---

## 2. Kaiming init: başlangıçta bir kez düzeltmek

**Bu bölüm neden var:** Rastgele başlatmanın bir ölçeği var ve o ölçek keyfi değil — varyans
hesabından çıkıyor.

### Neden $\sqrt{\text{fan\_in}}$'e bölünüyor

`z = x @ W`'nin tek bir elemanı, `fan_in` tane çarpımın toplamı:

$$z_j = \sum_{k=1}^{\text{fan\_in}} x_k W_{kj}$$

Bağımsız terimlerin toplamının varyansı, varyansların toplamı
(bkz. [07 · Neden Gradient Descent?](/posts/yz50-07-neden-gradient-descent/) notunun `Var` türetimi):

$$\mathrm{Var}[z] = \text{fan\_in} \cdot \mathrm{Var}[x]\,\mathrm{Var}[W]$$

Girdinin varyansını korumak istiyorsan ($\mathrm{Var}[z] = \mathrm{Var}[x]$):

$$\mathrm{Var}[W] = \frac{1}{\text{fan\_in}} \quad\Longrightarrow\quad \text{std}(W) = \frac{1}{\sqrt{\text{fan\_in}}}$$

**`fan_in` = o layer'a gelen girdi sayısı.** Bölme buradan geliyor: layer ne kadar genişse, her
ağırlık o kadar küçük olmalı ki toplam aynı ölçekte kalsın.

### `gain` çarpanı neden var

Yukarıdaki hesap aktivasyon fonksiyonunu hiç hesaba katmıyor. Ama `tanh` çıktıyı **sıkıştırıyor** —
varyansı azaltıyor. Bunu telafi etmek için bir `gain` çarpanı ekleniyor:

$$\text{std}(W) = \frac{\text{gain}}{\sqrt{\text{fan\_in}}}$$

| Aktivasyon | gain | sebep |
|---|---|---|
| kimlik (aktivasyon yok) | 1 | sıkıştırma yok |
| `tanh` | **5/3 ≈ 1.67** | tanh varyansı azaltıyor, telafi gerekiyor |
| `ReLU` | $\sqrt{2}$ | negatif yarıyı sıfırladığı için varyansın yarısını atıyor |

`ReLU`'nun $\sqrt{2}$'si He ve ark. 2015'in asıl katkısı — "Kaiming init" adı oradan geliyor
(Kaiming He). `tanh` için 5/3 ise Xavier/Glorot geleneğinden.

### Kendi kodunda

`mlp-2.py:80`:

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
```

`(5/3)` gain, `(n_embd * block_size)` ise `fan_in` — 3 harf × 10 boyutlu embedding = 30.
PyTorchified halinde (`mlp-2.py:194`) gain 1.0'a düşüyor, çünkü orada BatchNorm devrede.

---

## 3. BatchNorm: eğitim boyunca sürekli düzeltmek

**Bu bölüm neden var:** Kaiming init ölçeği **başlangıçta** doğru ayarlıyor. Ama eğitim ilerledikçe
ağırlıklar değişiyor ve ölçek yeniden kayıyor. BatchNorm bu kaymayı her adımda geri alıyor.

### Ne yapıyor

Her layer'ın aktivasyon-öncesi çıktısını, **o anki minibatch'in istatistiğiyle** normalize ediyor:

$$
\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \qquad
\text{sonra} \qquad z_{\text{out}} = \gamma\,\hat{z} + \beta
$$

Birinci adım tanıdık: [01](/posts/yz50-01-lineer-regresyon/)'deki standartlaştırmanın aynısı — ama veriye
değil, **ağın ortasındaki aktivasyonlara** uygulanıyor ve her adımda tekrarlanıyor.

**$\gamma$ ve $\beta$ neden var:** normalize etmek layer'ın ifade gücünü kısıtlıyor — artık çıktı
hep ortalama 0, std 1 olmak zorunda. $\gamma$ (ölçek) ve $\beta$ (kaydırma) **öğrenilen**
parametreler; ağ isterse normalizasyonu kısmen geri alabiliyor. Yani BatchNorm "normalize et"
değil, **"ölçeği öğrenilebilir bir parametre yap"** demek.

### Eğitim ve tahmin farkı — en çok hata yapılan yer

| | Hangi istatistik | Neden |
|---|---|---|
| **Eğitim** | o anki minibatch'in $\mu$, $\sigma^2$'si | Batch'ler rastgele olduğu için bu bir gürültü kaynağı, hafif düzenleyici (regularizer) etkisi de var |
| **Tahmin** | eğitim boyunca biriktirilmiş **running** $\mu$, $\sigma^2$ | Tek bir örnek için "batch ortalaması" diye bir şey yok; ayrıca tahmin, hangi örneklerle birlikte geldiğine bağlı olmamalı |

İkisini karıştırmak sessiz bir hata: model eğitimde iyi görünür, tek örnekle tahmin yaparken
saçmalar.

### Kendi kodunda

`mlp-2.py:207`'deki `BatchNorm1d` sınıfı bunu elle yazıyor — `self.training` bayrağı, `momentum`
ile running istatistik güncellemesi, `gamma`/`beta` parametreleri. `torch.nn.BatchNorm1d`'in
yaptığının aynısı.

**Bir yan etki:** BatchNorm ortalamayı çıkardığı için, kendinden önceki layer'ın **bias'ı işlevsiz
kalıyor** — çıkarılan ortalamanın içinde eriyor. `mlp-2.py:81`'deki yorum bunu söylüyor: `b1`'e
gerek yok, işini `bnbias` görüyor.

---

## 4. Ölçülen sonuç — beklentiyi düşür

**Bu bölüm neden var:** Bu tekniklerin "loss'u ciddi düşürdüğü" beklentisi yaygın ve yanlış.
Kendi ölçümün ne dediğine bakmak gerekiyor.

`mlp-2.py`'deki yorumlardan, kendi koşunun sonuçları:

| Aşama | train | val |
|---|---|---|
| başlangıç | 2.1245 | 2.1682 |
| softmax'ın aşırı güvenini düzelt | 2.07 | 2.13 |
| tanh'ın init'te doymasını düzelt | 2.0356 | 2.1027 |
| Kaiming init | 2.0377 | 2.1070 |
| BatchNorm eklendi | 2.0668 | **2.1048** |

**Asıl sıçrama ilk iki init düzeltmesinden geliyor** (2.1682 → 2.1027). Kaiming ve BatchNorm bu
ölçekte val loss'u kayda değer biçimde iyileştirmiyor — hatta Kaiming tek başına biraz kötüleştirmiş
görünüyor (gürültü sınırında).

**O zaman ne işe yarıyorlar:**

1. **İnce ayar zorunluluğunu kaldırıyorlar.** Elle "bu layer için 0.2, şu layer için 0.05" diye
   uğraşmak yerine bir kural veriyorlar
2. **Derinlikte fark açılıyor.** 2 layer'da fark yok, 20 layer'da eğitimin çalışıp çalışmaması
   meselesi. Bu ölçüm 6 layer'lık bir ağda yapıldı, avantajın görünmeye başlamadığı bölge
3. **Daha büyük learning rate'e izin veriyorlar** — yani [01](/posts/yz50-01-lineer-regresyon/)'deki
   $2/L$ zincirinin tam karşılığı

---

## 5. Nereye bağlanıyor

**Bu bölüm neden var:** Bu iki teknik yalıtılmış numaralar değil, üç ayrı notun kesiştiği yer.

- [01 · Lineer Regresyon](/posts/yz50-01-lineer-regresyon/), bölüm 4 — $\mathrm{lr} < 2/L$ ve ölçeğin $L$'yi
  belirlemesi. Bu notun **gerekçesi** orada
- [13 · Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/) — tanh'ın doyması ve vanishing
  gradient. Bu notun **çözdüğü problem** orada
- [02 · Hiperuzay ve Matrix Gradient](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/), bölüm 5 — condition number ve
  standartlaştırma. Aynı fikrin iki parametreli, çizilebilir hali

**Sonraki adım:** BatchNorm'un backward'ını elle yazmak — YZ50 Hafta 5'in görevlerinden biri.
Orada `bnmeani → bndiff → bndiff2 → bnvar → bnvar_inv → bnraw` zinciri olarak parçalanıyor;
`.std()` çağrısı yerine varyans zinciri yazılmasının sebebi de türevin o zincirden geçmesi.

## Özet

1. Problem ölçeğin layer layer kayması; tek layer'da elle çözülür, derinlikte üstel olarak bozulur.
2. Kaiming init: `std(W) = gain / sqrt(fan_in)`. Bölme varyans toplamından, `gain` aktivasyonun
   sıkıştırmasını telafi etmekten geliyor (tanh 5/3, ReLU $\sqrt{2}$).
3. BatchNorm: aynı standartlaştırmayı ağın ortasında, her adımda yapıyor. $\gamma/\beta$ ile
   öğrenilebilir kalıyor.
4. Eğitim minibatch istatistiği, tahmin running istatistik kullanır — karıştırmak sessiz hata.
5. Bu ölçekte val loss'a katkıları küçük; asıl faydaları ince ayardan kurtarmak ve derinliği
   mümkün kılmak.
