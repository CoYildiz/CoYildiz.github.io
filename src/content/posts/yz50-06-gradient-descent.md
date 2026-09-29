---
title: "06 · Gradient Descent"
published: 2026-09-29
description: "Gradient descent'in geometrisi: negatif gradient neden en dik iniş yönü, cost yüzeyi nasıl bir şey ve öğrenme tam olarak nerede gerçekleşiyor."
tags:
  - YZ50
  - Gradient Descent
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

Kaynak: [3blue1brown.com/lessons/gradient-descent](https://www.3blue1brown.com/lessons/gradient-descent), orijinal video [YouTube'da](https://www.youtube.com/watch?v=IHZwWFHWa-w). Önceki: [04 · Sinir Ağı Nedir?](/posts/yz50-04-sinir-agi-nedir/).

---

## Geçen Videodan Hatırlatma + Yeni Bilgi

Ağırlıklı toplam + bias'tan sonra bir nonlineer fonksiyon uygulanıyordu — Video 1'de sadece **sigmoid** görmüştük, bu video ek olarak **ReLU**'yu da alternatif olarak anıyor (detaya girmiyor, sadece "sigmoid veya ReLU gibi" diyor). Not olarak düşsün, ileride karşımıza çıkabilir.

## Ağ Nasıl Öğrenir

**Bu bölüm neden var:** "Öğrenme" kelimesi bir şey anlatmıyor. Burada ne olduğu somutlaşıyor: elde etiketli veri var, model yanılıyor, ve yanılma miktarını azaltacak şekilde parametreler değiştiriliyor.

Diğer yazılım mühendisliğinden farkı: rakam tanıma için **elle bir algoritma yazmıyorsun**. Bunun yerine, etiketli örnekler (görüntü + doğru cevap) alıp 13.002 weight/bias'ı bu örneklerde daha iyi performans verecek şekilde ayarlayan bir algoritma yazıyorsun.

- **Training data:** Ağı eğitmek için kullanılan etiketli örnekler (bu video, [MNIST](http://yann.lecun.com/exdb/mnist/) veri setinden bahsediyor — on binlerce etiketli el yazısı rakam).
- **Test data:** Ağın hiç görmediği, eğitimden sonra "gerçekten genelleyebiliyor mu" diye kontrol etmek için ayrılan veri.

"Öğrenme" kelimesi gizemli gelebilir ama aslında **bir fonksiyonun minimumunu bulma** işi — bir kalkülüs egzersizi.

## Cost Function (Maliyet Fonksiyonu)

**Bu bölüm neden var:** "Model kötü" demek bir sayı değil. Optimizasyon yapabilmek için kötülüğü **tek bir sayıya** indirmek gerekiyor — cost function tam olarak bu. Hangi loss'un seçileceği ayrı bir konu ([16](/posts/yz50-16-loss-nll-cross-entropy/)).

Ağın weight/bias'ları başta **rastgele** — yani başta çöp gibi çalışıyor. "Ne kadar kötü" olduğunu sayısal olarak ölçmek için bir **cost function** (maliyet fonksiyonu) tanımlanır.

**Tek bir eğitim örneği için:** çıktı layer'ındaki her nöronun activation'ı ile "olması gereken" değer (doğru rakam için 1, diğerleri için 0) arasındaki farkların karelerinin toplamı:

$$C = \sum_{i=0}^{9} (a_i - y_i)^2$$

- $a_i$: ağın verdiği activation, $y_i$: istenen değer (0 veya 1)
- Ağ doğru rakamdan eminse → cost küçük. Yanlışsa/kararsızsa → cost büyük.

*(Bu, Görev 3'ün "basit bir loss fonksiyonu" tam olarak istediği formül — kodunu ben yazmıyorum, bu senin tasarlayacağın kısım.)*

## Tüm Örnekler Üzerinden Ortalama Maliyet

Tek görüntü değil, **on binlerce eğitim örneğinin ortalama cost'u** asıl ölçüt. Cost function'ı bir üst layer gibi düşün:

- **Sinir ağının kendisi** bir fonksiyon: 784 girdi → 10 çıktı, 13.002 parametre.
- **Cost function** onun üstüne bir layer daha: girdisi o 13.002 weight/bias, çıktısı **tek bir sayı** (ne kadar kötü), tüm training data'ya göre hesaplanıyor.

## Maliyet Fonksiyonunu Minimize Etmek — Gradient Descent Sezgisi

**Bu bölüm neden var:** Cost'u tanımlamak yetmiyor, onu küçültmek gerekiyor. Neden analitik olarak çözmek yerine adım adım iniyoruz — bunun cevabı [07](/posts/yz50-07-neden-gradient-descent/)'de, burada önce sezgisi kuruluyor.

Önce basitleştirelim: 13.002 girdili değil, **1 girdili, 1 çıktılı** bir cost function hayal et. Minimumunu nasıl bulursun?

- Kalkülüs dersinden bilinen "eğim (slope) = 0" noktasını analitik çözmek, karmaşık fonksiyonlarda (bizimki gibi) pratik değil.
- Daha esnek yöntem: **rastgele bir noktadan başla**, o noktadaki eğime bak. Eğim negatifse sağa, pozitifse sola kay. Tekrarla.
- **Adım büyüklüğünü eğimle orantılı yap** — minimuma yaklaştıkça eğim düzleşir, adımlar otomatik küçülür, bu da minimumu "geçmeni" (overshoot) engeller.
- Görsel: bir topun tepeden aşağı yuvarlanması.
- **Önemli uyarı:** bulduğun **local minimum**, illa **global minimum** (en küçük olası değer) olmak zorunda değil — başladığın rastgele noktaya bağlı. Garanti yok.

![Top tepeden yuvarlanıyor — local vs global minimum](/yz50/top-yuvarlanmasi.svg)

*(Görev 4 — "parametreleri manuel değiştirerek loss'un nasıl değiştiğini gözlemle, loss eğrisini çiz" — tam olarak bu bölümün görselini senin elinle çizmen: 1 parametreli basit bir cost function için topun-tepeden-yuvarlanması eğrisi.)*

## Gradient — Eğimin Çok Boyutluya Genellemesi

**Bu bölüm neden var:** Tek değişkende "eğim" tanıdık. 13.002 boyutta eğimin ne demek olduğu tanıdık değil — gradient o genelleme, ve neden **vektör** olduğu burada çıkıyor.

2+ girdili bir fonksiyonda "eğim" tek bir sayı olamaz — yön de önemli. Bunun yerine bir **vektör** kullanılır: **gradient** (∇C).

- Gradient, fonksiyonu **en hızlı artıracak** yönü gösterir (steepest ascent).
- **Negatif gradient** ($-\nabla C$), fonksiyonu en hızlı **azaltacak** yönü gösterir — bizim istediğimiz bu.
- Gradient vektörünün **büyüklüğü**, o yöndeki eğimin ne kadar dik olduğunu gösterir.

**Gradient descent algoritması:** gradient'i hesapla → negatif gradient yönünde küçük bir adım at → tekrarla.

Her adımda güncelleme (kavramsal olarak):

$$\vec{w} \leftarrow \vec{w} - \eta \nabla C$$

η ("eta") = **learning rate** (öğrenme oranı) — adım büyüklüğünü ölçekler. Büyük η → daha hızlı ama minimumu geçip etrafında salınma (oscillation) riski.

## Gradient'i Başka Türlü Düşünmek

13.002 weight/bias'ı tek bir dev vektörde topla. Negatif gradient da aynı boyutta bir vektör — her bir weight/bias'a "ne kadar ve hangi yönde dokunulması gerektiğini" söyleyen bir **nudge (dürtme) listesi**:

- **İşaret (+/-):** o parametre artırılmalı mı azaltılmalı mı
- **Büyüklük:** o parametrenin cost'u ne kadar etkilediği — yani hangi değişikliklerin "daha çok işe yaradığı" (bazı bağlantılar diğerlerinden daha önemli)

Bu gradient'i **verimli şekilde hesaplama algoritmasının adı: backpropagation** — bir sonraki derste (Karpathy'nin videosu / YZ50'nin ileriki haftaları) detaylı işlenecek.

**Bu videonun özeti tek cümlede:** Ağın "öğrenmesi" = cost function'ı minimize edecek şekilde weight/bias'ları ayarlamak, ve bunun yöntemi gradient descent.

---

## Bu Videonun Göreve Bağlantısı

- **Görev 3** (basit loss fonksiyonu) ↔ "Cost Function" bölümü — formül yukarıda, kodu sen yazacaksın.
- **Görev 4** (parametre/loss gözlemi + eğri çizimi) ↔ "Minimize Etmek" bölümündeki 1D top-yuvarlanma görseli — bunu kendi basit örneğinle üretmen isteniyor.
- **Görev 5** (numerical derivativele gradient descent) ↔ bu video **kavramsal temeli** (gradient, negatif gradient yönünde adım atma, learning rate) veriyor; **numerical derivative'in nasıl hesaplanacağı** (finite difference: (f(x+h)-f(x)) / h) Karpathy'nin videosunda (Video 3) geliyor — oraya bakınca bu notlara ek yapılacak.

**Bağlantılı:** [04 · Sinir Ağı Nedir?](/posts/yz50-04-sinir-agi-nedir/)
