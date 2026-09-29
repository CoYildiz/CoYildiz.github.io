---
title: "15 · Aktivasyon + Loss Eşleşmesi"
published: 2026-09-29
description: "Sigmoid+BCE ve softmax+cross-entropy'de türevin neden hep (ŷ−y)'ye sadeleştiği — tesadüf değil, GLM'den gelen yapısal bir sonuç."
tags:
  - YZ50
  - Aktivasyon
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

Kaynak: kullanıcı sorusu (2026-08-28) — sigmoid+BCE (Binary Cross-Entropy)'deki sadeleşmenin "daha detaylı" hali istendi.

---

## 1. Sigmoid + Binary Cross-Entropy — Adım Adım Tam İspat

**Bu bölüm neden var:** [03](/posts/yz50-03-siniflandirmaya-gecis/)'te "sadeleşiyor" denip geçilmişti. Burada tam ispat yapılıyor, çünkü sadeleşmenin **neden** olduğu tesadüf olmadığını ancak adımları görünce anlıyorsun.

Loss (tek örnek için):

$$L = -\left[y\ln(a) + (1-y)\ln(1-a)\right], \qquad a = \sigma(z) = \frac{1}{1+e^{-z}}$$

**Adım 1 — $\partial L/\partial a$:**

$$\frac{\partial L}{\partial a} = -\left[\frac{y}{a} - \frac{1-y}{1-a}\right] = \frac{-y(1-a) + (1-y)a}{a(1-a)} = \frac{a-y}{a(1-a)}$$

**Adım 2 — $\partial a/\partial z$:** bu zaten bildiğin sigmoid derivative'i:

$$\frac{\partial a}{\partial z} = \sigma'(z) = a(1-a)$$

**Adım 3 — chain rule:**

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = \frac{a-y}{a(1-a)} \cdot a(1-a) = a-y$$

**a(1-a) terimleri birebir sadeleşiyor.** Geriye sadece `(a-y)` kalıyor — bugün bütün gün konuştuğumuz saturasyon/vanishing-gradient teriminin **hiçbir izi yok.**

**Karşılaştır — quadratic cost'la aynı hesap:** $C=(a-y)^2$ kullansaydın, $\partial C/\partial z = 2(a-y)\sigma'(z) = 2(a-y)a(1-a)$ — burada `a(1-a)` **kalıyor**, doymuşsa (`a` neredeyse 0 veya 1) bu terim sıfıra gidip gradient'i öldürüyor. Sadeleşme sadece cross-entropy ile olduğunda gerçekleşiyor.

---

## 2. Softmax + Categorical Cross-Entropy — Genellemesi (Daha Zor, Tam İspat)

**Bu bölüm neden var:** İkili halde sadeleşme şanstan ibaret görünebilir. Çok sınıflı halde de aynı sonucun çıkması, altta bir kural olduğunun kanıtı — ve bu, YZ50 Hafta 5'te `dlogits`'i elle yazarken karşına çıkacak ifadenin ta kendisi.

K sınıflı bir çıktı düşün: $z_1,\dots,z_K$ (ham skorlar), softmax: $a_i = e^{z_i} / \sum_k e^{z_k}$. Etiket $y$ one-hot (doğru sınıf $c$ için $y_c=1$, diğerleri 0). Loss: $L = -\sum_i y_i\ln(a_i) = -\ln(a_c)$.

**Zorluk:** sigmoid'den farklı olarak, her $a_i$ **tüm** $z_j$'lere bağlı (ortak payda yüzünden) — yani $\partial L/\partial z_j$ hesaplarken **tüm** $a_i$'lerin katkısını toplamak gerekiyor (bu, [12](/posts/yz50-12-backpropagation-calculus/)'daki "çoklu yol etkisi" toplamının aynısı).

**Softmax'ın kendi derivative'i (Jacobian):**

$$\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j), \qquad \delta_{ij} = \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}$$

**Chain rule, tüm i'ler üzerinden toplayarak** ($\partial L/\partial a_i = -y_i/a_i$, sadece $i=c$'de sıfırdan farklı):

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial a_i}\frac{\partial a_i}{\partial z_j} = \sum_i \left[-\frac{y_i}{a_i}\right]\left[a_i(\delta_{ij}-a_j)\right] = -\sum_i y_i\delta_{ij} + a_j\sum_i y_i$$

$\sum_i y_i\delta_{ij} = y_j$ (delta sadece i=j'de 1 bırakıyor), ve $\sum_i y_i = 1$ (one-hot, toplamı 1):

$$\frac{\partial L}{\partial z_j} = -y_j + a_j = a_j - y_j$$

**Aynı sonuç — $a_j - y_j$.** Softmax+CCE, sigmoid+BCE'nin çok-sınıflı genellemesi, aynı sadeleşme (daha karmaşık bir ispatla, çünkü softmax'ın kendisi çok-girdili/çok-çıktılı bir fonksiyon).

---

## 3. Regresyon — Linear + MSE (En Basit Hali)

**Bu bölüm neden var:** Üçüncü örnek, ilk ikisinin özel hali gibi görünmeyecek kadar farklı bir yerden geliyor (aktivasyon yok, sınıflandırma yok). Aynı sonucun buradan da çıkması, deseni tesadüf olmaktan çıkarıyor.

Çıktı layer'ında aktivasyon yoksa ($a=z$, "linear activation" — identity fonksiyon), loss $L = \frac{1}{2}(a-y)^2$:

$$\frac{\partial L}{\partial a} = a-y, \qquad \frac{\partial a}{\partial z} = 1, \qquad \frac{\partial L}{\partial z} = (a-y)\cdot 1 = a-y$$

Yine $(a-y)$. Burada "sadeleşecek" bir şey bile yok çünkü $\partial a/\partial z=1$ zaten en baştan basit — ama **aynı genel deseni** izliyor.

---

## 4. Asıl Soru — Bu Üçü Neden Hep Aynı Şeye Çıkıyor?

**Bu bölüm neden var:** Üç örnek bir teorem değil. Ortak sebebi adlandırmadan "hep böyle oluyor" demek ezber olur — bu bölüm sebebi veriyor.

Bu bir tesadüf değil — **tek bir genel teoremin üç örneği.** Kaynağı istatistikte **Generalized Linear Models (GLM)** teorisi.

**Fikir:** `y`'nin geldiği dağılım (Bernoulli — ikili sınıflandırma, Categorical — çok sınıflı, Gaussian — regresyon) **üstel aile**'den (exponential family) bir dağılım. Her üstel aile dağılımının kendine ait bir **"canonical link" (kanonik bağlantı) fonksiyonu** var — bu, ham skor `z`'yi doğrudan dağılımın "doğal parametresi" yapan özel bir seçim.

**Teorem (GLM teorisinden):** canonical link seçildiğinde, negatif log-likelihood'un (yani loss'un) `z`'ye göre derivative'i **her zaman ve otomatik olarak** `(a-y)` çıkıyor — dağılım ne olursa olsun.

![Üç eşleşme: ikisi doğru, biri yanlış](/yz50/kanonik-eslesme.png)

İlk iki panel kanonik eşleşme: gradient `z` büyüdükçe **hatayla birlikte büyüyor**, doyma yok.
Üçüncü panel aynı aktivasyonun yanlış loss'la eşleşmesi — `σ′` sadeleşmediği için hayatta kalıyor
ve gradient her iki uçta da sıfıra gidiyor. Teoremin pratikteki anlamı bu: doğru eşleşme, derivative'ten
aktivasyonun derivative'ini **siliyor**.

**Üç örnek, üç canonical link:**

| Dağılım | Canonical link (aktivasyon) | Negatif log-likelihood (loss) |
|---|---|---|
| Bernoulli (ikili) | sigmoid | binary cross-entropy |
| Categorical (çok sınıf) | softmax | categorical cross-entropy |
| Gaussian (sürekli) | identity (linear) | (sabit varyansla) kareler toplamı, yani MSE |

**Yani sigmoid'in "doğru" aktivasyon olması bir tercih değil — Bernoulli dağılımının canonical link'i matematiksel olarak sigmoid'in ta kendisi.** Softmax, Categorical dağılımının canonical link'i. Linear (aktivasyonsuz), Gaussian'ın canonical link'i. Cross-entropy/MSE de bu dağılımların kendi negatif log-likelihood'ları — hepsi zaten "doğal" bir çift olarak geliyor, insanlar "bu ikisini birlikte kullanalım" diye keyfi karar vermedi.

**Bunun ML mühendisliği için anlamı:** bir çıktı layer'ı tasarlarken "çıktım hangi dağılımdan geliyor" sorusunu sor (ikili mi, çok-sınıflı mı, sürekli mi) — cevap otomatik olarak hangi aktivasyon+loss çiftini kullanman gerektiğini söylüyor. Bu, ezberlenecek bir tablo değil, **çıkarılabilir** bir sonuç.

---

## Bağlantı

Bu, senin şu anki `Value`/`MLP (Multi-Layer Perceptron)` görevinin **dışında** — output layer'ı tasarımı, Faz 3-4'te PyTorch'la gerçek sınıflandırma/regresyon yaparken (`nn.BCELoss`, `nn.CrossEntropyLoss`, `nn.MSELoss`) doğrudan karşına çıkacak. Hidden layer'daki tanh seçimin ayrı bir konu (bkz. [13](/posts/yz50-13-aktivasyon-fonksiyonlari/)).

**Bağlantılı:** [13 · Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/) · [12 · Backpropagation Calculus](/posts/yz50-12-backpropagation-calculus/).
