---
title: "Hafta 2 — Optimizer Varyantları: Momentum, AdaGrad, RMSprop, Adam"
published: 2026-09-29
description: "Momentum, AdaGrad, RMSprop ve Adam'ın her birinin çözdüğü problem — epsilon'un karekökün dışında olması ve bias correction dahil."
tags:
  - YZ50
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

Kaynak: bir blog yazısı (kullanıcı yapıştırdı). **Formüller 2026-09-20'de birincil kaynaklara karşı yeniden doğrulandı** — aşağıdaki düzeltme kutusuna bak. Birincil kaynaklar: AdaGrad için Duchi ve ark. 2011, RMSprop için Tieleman & Hinton ders notları, Adam için [Kingma & Ba 2015](https://arxiv.org/abs/1412.6980), ve uygulama tarafı için [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) dokümantasyonu.

**Önceki nottan ([08](/posts/yz50-08-stochastic-gradient-descent/)) farkı:** o not "hangi veriyi örnekleyelim" sorusuna (mini-batch seçimi) cevap veriyordu. Bu not farklı bir eksen — **gradient'i elde ettikten sonra onu nasıl kullanıyoruz**, adımı nasıl daha akıllı atıyoruz. İkisi birbirini dışlamıyor, üst üste biniyor (Adam da mini-batch'ler üzerinde çalışır).

**⚠️ Durum (2026-08-27 netleşti): yüzeysel/geçici okuma.** Bu notun var olması, Faz 4'ün PyTorch/optimizer konularının yapıldığı anlamına gelmiyor — derinlemesine çalışma (gerçek kodla `torch.optim` kullanmak, hyperparameter'larla deney yapmak) Faz 4'e kalıyor.

---

> **⚠️ Düzeltme (2026-09-20) — epsilon'un yeri yanlıştı.** Bu notun önceki hali üç formülde de
> `√(v + ε)` yazıyordu, yani epsilon'u **karekökün içine** koyuyordu. Doğrusu **dışında**:
> `√v + ε`. [PyTorch dokümantasyonundan](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
> birebir: *θₜ ← θₜ₋₁ − γ·m̂ₜ/(√v̂ₜ + ε)*.
>
> **Neden önemli:** epsilon'un işi paydanın sıfır olmasını engellemek. Karekökün **içinde**
> olursa `v` zaten çok küçükken katkısı `√ε ≈ 1e-4` gibi bir şey olur ve adımı gereksiz yere
> kısar; **dışında** olduğunda katkısı doğrudan `ε = 1e-8` kalır. Küçük bir detay gibi görünüyor
> ama `v → 0` olduğunda ikisi arasında **dört mertebe** fark var.
>
> *(RMSprop'un bazı eski anlatımlarında epsilon gerçekten karekökün içinde yazılıyor — ama
> `torch.optim.RMSprop` dışında kullanıyor. Kütüphaneyle uyumlu hali yazıldı.)*

## Neden Var — Hessian notuyla bağlantı

Birkaç mesaj önce konuştuğumuz "ball-mimicking gradient descent, second derivative (Hessian) gerektirir ama bu O(n^2) ve pahalı" pasajını hatırla. Aşağıdaki yöntemlerin hepsi, **gerçek second derivative'i hiç hesaplamadan**, geçmiş gradient'lerin bir özetini tutarak eğriliği "taklit eden" ucuz numaralar — tam olarak kitabın bahsettiği "kaçış yolları".

## Momentum

**Bu bölüm neden var:** Düz gradient descent her adımı sıfırdan değerlendiriyor — geçmişi hatırlamıyor. Momentum, "aynı yöne ısrarla gidiyorsan hızlan" fikrinin en basit hali ve sonraki üç yöntemin de yarısı.

```
v(t) = γ·v(t-1) + α·∇J(θ)
θ = θ − v(t)
```

Gradient'i doğrudan kullanmak yerine, bir **hız (velocity)** biriktiriyorsun — geçmiş adımların ağırlıklı toplamı (γ≈0.9 tipik). Sezgi: gerçek bir topun ataleti gibi.

- Gradient **sürekli aynı yöndeyse** → hız birikir, adım büyür → hızlanma
- Gradient **yön değiştirip duruyorsa** (dar bir vadide zikzak) → hız bunu yumuşatır, salınımı azaltır
- Düz/yayvan bölgelerde (plateau) → birikmiş hız seni ileri itmeye devam eder, saplanıp kalmazsın

**Dar/uzun bir vadide gerçek bir simülasyon** ($\theta_1$ yayvan yön, $\theta_2$ dik yön, hedef orijinde):

![Momentum vs düz gradient descent](/yz50/momentum-vs-gd.png)

Soldaki (düz GD), dik yönde ($\theta_2$) tek bir sıçramayla hemen sakinleşiyor — ama yayvan yönde ($\theta_1$) acı verecek kadar yavaş, 55 adım sonunda hâlâ hedeften çok uzakta. Sağdaki (momentum) başta daha belirgin salınıyor, ama **her salınım bir öncekinden küçük** (sönümleniyor) ve aynı zamanda yayvan yönde hız biriktirip gerçekten hedefe ulaşıyor. Asıl kazanç "daha az zıplama" değil — **toplam yakınsama hızı**: momentum, ısrarla aynı yöne işaret eden gradient'i (θ_1 ekseni) biriktirip hızlanabiliyor, düz GD ise her adımı sıfırdan değerlendirdiği için o birikimi hiç yapamıyor.

## Nesterov Accelerated Gradient (NAG)

**Bu bölüm neden var:** Momentum'un tek bir kusuru var: gideceği yeri bilmesine rağmen gradient'i hâlâ durduğu yerde ölçüyor. NAG bu küçük farkı düzeltiyor — küçük ama teorik yakınsama oranını değiştiren bir fark.

```
v(t) = γ·v(t-1) + α·∇J(θ − γ·v(t-1))
θ = θ − v(t)
```

Momentum'un küçük bir iyileştirmesi: gradient'i **şu anki** $\theta$'da değil, "hızın seni zaten götüreceği" ileri noktada hesaplıyorsun ("look-ahead"). Sezgi: köşeye yaklaşan bir topun, çarpmadan önce yavaşlamaya başlaması gibi.

## AdaGrad — Parametre Başına Uyarlanabilir Learning Rate

**Bu bölüm neden var:** Buraya kadar tek bir `lr` bütün parametreler için aynıydı. Ama [01](/posts/yz50-01-lineer-regresyon/)'de gördüğün gibi farklı yönlerin eğriliği farklı — AdaGrad her parametreye kendi adım boyunu veriyor.

```
G(t) = G(t-1) + ∇J(θ)^2
θ = θ − α · ∇J(θ) / (√G(t) + ε)
```

Her parametrenin **kendi** learning rate'i var — G(t), o parametrenin gradient'inın karelerinin **toplamı** (zamanla sürekli büyüyor). Sık/büyük gradient alan parametrelerin adımı küçülür, seyrek/küçük gradient alanların adımı büyür.

**Zayıflık:** G(t) sürekli arttığı için (hiç azalmıyor) learning rate zamanla sıfıra yaklaşır — ağ bir süre sonra öğrenmeyi neredeyse durdurur.

## RMSprop — AdaGrad'ın Zayıflığını Düzeltir

**Bu bölüm neden var:** AdaGrad'ın biriktirdiği toplam hiç azalmıyor, dolayısıyla learning rate kaçınılmaz olarak sıfıra gidiyor. RMSprop tek bir değişiklikle (toplam yerine hareketli ortalama) bunu çözüyor.

```
E[g^2](t) = γ·E[g^2](t-1) + (1−γ)·∇J(θ)^2
θ = θ − α · ∇J(θ) / (√E[g^2](t) + ε)
```

AdaGrad'daki "sürekli toplam" yerine **üstel hareketli ortalama** (exponential moving average) kullanıyor — eski gradient'ler zamanla unutuluyor, birikim sonsuza gitmiyor. Sonuç: learning rate küçülüyor ama AdaGrad gibi "ölmüyor".

## Adam — Yaygın Varsayılan Tercih

**Bu bölüm neden var:** Adam yeni bir fikir getirmiyor — momentum ile RMSprop'u birleştiriyor. Pratikte varsayılan tercih olduğu için içinde ne olduğunu bilmek gerekiyor, özellikle de `torch.optim.Adam` yazdığında.

```
m(t) = β₁·m(t-1) + (1−β₁)·∇J(θ)          (momentum kısmı)
v(t) = β₂·v(t-1) + (1−β₂)·∇J(θ)^2         (RMSprop kısmı)

m̂ = m(t) / (1 − β₁^t)                     (bias correction)
v̂ = v(t) / (1 − β₂^t)

θ = θ − α · m̂ / (√v̂ + ε)
```

Momentum (m, yön/hız) ile RMSprop'u (v, parametre-başına ölçekleme) birleştiriyor.

**Bias correction neden var:** `m` ve `v` sıfırdan başlıyor, dolayısıyla ilk adımlarda gerçek
değerlerinden **küçük** kalıyorlar (sıfıra doğru yanlı). `1 − β^t` ile bölmek bunu düzeltiyor:
`t=1`'de bölen `1−β₁ = 0.1` olduğu için `m̂` on katına çıkıyor, `t` büyüdükçe bölen 1'e yaklaşıyor
ve düzeltme kendiliğinden devreden çıkıyor. Bu olmadan Adam ilk birkaç yüz adımda gereğinden
yavaş ilerliyor.

**PyTorch varsayılanları:** `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

**Pratikte:** çoğu problemde makul bir varsayılan — `torch.optim.Adam` olarak Faz 4'te doğrudan kullanacaksın. (Bazı araştırma bağlamlarında, ince ayar/genelleme performansı için düz SGD (Stochastic Gradient Descent)+momentum hâlâ tercih ediliyor — ama başlangıç için Adam güvenli seçim.)

## Bonus — Bir Kez Duyman Yeterli Terimler

- **Learning rate scheduling:** öğrenme oranını ($\alpha$) zamanla azaltmak (exponential decay, cosine annealing) — eğitimin başında büyük adım, sona doğru küçük/hassas adım.
- **Gradient clipping:** gradient çok büyürse ("patlarsa") normunu bir tavana kırpmak — kararsız güncellemeleri önler.
- **Xavier/He initialization:** weight'leri rastgele ama **belirli bir varyansla** başlatmak — Faz 4'te göreceksin, `Value`/`MLP (Multi-Layer Perceptron)`'ni şu an muhtemelen basit rastgele başlatmayla kuruyorsun, büyük ağlarda bu seçim önemli hale geliyor.
- **Batch normalization:** her mini-batch'in istatistiklerini (ortalama/varyans) layer aktivasyonlarını normalize etmek için kullanmak — CNN (Convolutional Neural Network)'lerde (Faz 4) çok yaygın.

---

## Bağlantı

Bunların hiçbiri Hafta 2'nin görevi değil — Görev 5'teki eğitim döngün muhtemelen düz gradient descent (`param -= lr * param.grad` tarzı, momentum/Adam yok). Ama Faz 4'te PyTorch'a geçince `torch.optim.SGD`, `torch.optim.Adam` gibi hazır sınıflar tam olarak yukarıdaki formülleri uyguluyor — o zaman bu not işine yarayacak.

**Bağlantılı:** [Hafta 2 — Stochastic Gradient Descent: Ne Zaman, Neden Rastgele, Alternatifler](/posts/yz50-08-stochastic-gradient-descent/).
