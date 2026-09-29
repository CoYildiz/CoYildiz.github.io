---
title: "Hafta 2 — Stochastic Gradient Descent: Ne Zaman, Neden Rastgele, Alternatifler"
published: 2026-09-29
description: "Tek örnekle adım atmak neden çalışıyor: batch boyutunun gürültü-hız dengesi, rastgeleliğin rolü ve alternatiflerle karşılaştırma."
tags:
  - YZ50
  - Gradient Descent
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

Kaynak: Michael Nielsen — [Neural Networks and Deep Learning, Bölüm 1](http://neuralnetworksanddeeplearning.com/chap1.html) (kullanıcı yapıştırdı), + genel ML bilgisi (Andrew Ng'nin batch GD ile başlayan sıralamasıyla karşılaştırınca kafa karıştırdığı için genişletildi).

---

## Orijinal Pasaj — SGD'nin Gerekçesi (Nielsen, Bölüm 1)

**Problem:** gerçek gradient $\nabla C$ = tüm $n$ eğitim örneğinin ortalaması. $n$ büyükse (MNIST'te 60.000), tek bir weight güncellemesi için 60.000 örneğin hepsini işlemek gerekiyor — yavaş.

**Fikir:** tüm veri yerine rastgele küçük bir mini-batch (m örnek) al, sadece onların gradient'inı hesapla, ortalamasını gerçek gradient'in **tahmini** olarak kullan: $\nabla C \approx \frac{1}{m}\sum_j \nabla C_{X_j}$. Gerekçesi istatistiksel — **anket analojisi**: bütün seçmen kitlesine sormak yerine rastgele bir örnekleme sormak, sonucu makul bir doğrulukla tahmin eder.

**Sayısal örnek:** n=60.000, m=10 → **6.000 kat hızlanma**. Tahmin mükemmel değil (istatistiksel dalgalanma var) ama olması da gerekmiyor — tek istenen C'yi azaltacak genel yön, kesin gradient değil.

![Minibatch boyutu ile standart hatanın azalması](/yz50/minibatch-varyans.png)

**Ölçekleme notu:** bazı kaynaklar 1/n veya 1/m çarpanını atlıyor — bu, öğrenme oranı η'yı yeniden ölçeklemekle aynı etkiyi yapıyor, kavramsal fark yaratmıyor ama farklı kaynakları karşılaştırırken dikkat edilmesi gereken bir detay.

### Egzersiz — Online Learning (m=1) vs Mini-batch (m=20)

**Soru:** online learning'in (her tek örnekten sonra güncelleme) mini-batch=20'ye göre bir avantajı ve bir dezavantajını söyle.

- **Avantaj:** her örnekten hemen sonra güncelleme yapılıyor, bir batch dolmasını beklemeye gerek yok — verinin gerçek zamanlı/akış halinde geldiği durumlarda (toplam örnek sayısı önceden bilinmiyorsa) değerli, ağ anında uyum sağlıyor.
- **Dezavantaj:** gradient tahmini çok gürültülü — tek örneğin gradient'i genel eğilimi değil, o örneğe özgü tuhaflığı yansıtır. m=20 bu gürültüyü ortalayarak yumuşatıyor; online learning'in güncellemeleri daha sık ama daha kararsız/sıçramalı.

---

## Üç Yöntem — Kafa Karışıklığının Kaynağı

**Bu bölüm neden var:** Batch GD, mini-batch GD ve SGD isimleri literatürde tutarsız kullanılıyor; aynı kelime farklı kaynaklarda farklı şeyi gösteriyor. Üçünü tek yerde ayırmak gerekiyor.

| | m (her adımda kaç örnek) | Gradient | Hız/adım | Gürültü |
|---|---|---|---|---|
| **Batch GD** | Tüm veri (n) | Kesin/tam | Yavaş | Yok |
| **Mini-batch SGD** | Küçük (örn. 10-256) | Tahmini | Hızlı | Orta |
| **Online/Incremental (SGD, m=1)** | 1 | Çok kaba tahmin | Çok hızlı/adım | Yüksek |

**Andrew Ng ile çelişki yok, sıralama farkı:** Stanford/Coursera'nın giriş dersleri genelde batch GD'yi önce öğretir (kavramsal olarak temiz, gürültü yok). Mini-batch, genelde daha ileri/pratik bir bölümde geliyor. Nielsen erken getiriyor çünkü MNIST (60.000 örnek) zaten batch GD'yi yavaşlatacak kadar büyük. **Pratikte:** modern derin öğrenmede (PyTorch, Faz 4) mini-batch SGD (veya Adam/RMSprop gibi üzerine kurulu derivative'leri) standart — full-batch GD sadece küçük/toy veri setlerinde kullanılır.

## Ne Zaman Mini-Batch Kullanılmalı

**Bu bölüm neden var:** "Mini-batch her zaman daha iyi" doğru değil. Hangi koşulda hangisinin kullanıldığı, veri büyüklüğüne ve donanıma bağlı bir karar.

- Veri seti, her adımda tamamını işlemek pratik olmayacak kadar büyükse (MNIST'in 60.000'i bile bunun için yeterince büyük örnek).
- Pratikte: gerçek dünya derin öğrenmesinde neredeyse **her zaman** — full-batch GD sadece çok küçük veri setlerinde ya da öğretim amaçlı basit örneklerde mantıklı.
- Batch boyutu (m) ayrıca donanım kısıtına da bağlı — GPU belleğine sığması gerekiyor, bu yüzden 32/64/128/256 gibi "2'nin katı" boyutlar yaygın (donanımsal verimlilik nedeniyle, matematiksel bir zorunluluk değil).

## Neden Rastgele Seçmeliyiz

**Bu bölüm neden var:** Rastgelelik bir detay gibi görünüyor ama iki istatistiksel özelliği taşıyor — yansızlık ve varyans. İkisinin ispatı [07](/posts/yz50-07-neden-gradient-descent/)'de; burada sonucu ve pratik karşılığı.

1. **Yansız (unbiased) tahmin şartı:** mini-batch'in ortalama gradient'inın gerçek gradient'e yakın olmasının matematiksel garantisi, örneklemin **rastgele** olmasına dayanıyor. Rastgele olmayan bir seçim (örn. veri setinin ilk 20 örneğini almak) sistematik bir sapmaya (bias) yol açabilir — mesela MNIST etikete göre sıralıysa, "ilk 20 örnek" tamamen aynı rakam olabilir, gradient o zaman tüm veri setini değil sadece o rakamı temsil eder.
2. **Optimizasyon yan faydası:** rastgele örneklemin getirdiği "gürültü", gradient descent'in sığ local minimum'lara veya düz/yayvan bölgelere (plateau) takılıp kalmasını zorlaştırabiliyor — kesin/pürüzsüz batch GD orada saplanıp kalabilirken, SGD'nin gürültüsü onu oradan "iter".
3. **Sabit sıra ezberlemeyi önlemek:** her epoch aynı sırayla giderse, ağ o sıraya özgü bir örüntüyü ezberleyebilir; rastgelelik (özellikle her epoch başında yeniden karıştırma) bunu engelliyor.

## Alternatif Yöntemler

**Bu bölüm neden var:** Saf rastgele seçim gerçekte pek kullanılmıyor. Kütüphanelerin (`DataLoader(shuffle=True)`) ne yaptığını bilmek gerekiyor, çünkü senin yazdığın kodla onun yaptığı aynı şey değil.

- **Shuffle-then-sequential (gerçekte en yaygın uygulama):** saf "her adımda bağımsız rastgele seç" yerine (bu bazı örnekleri atlayabilir, bazılarını tekrar seçebilir), pratikte şu yapılır: **her epoch başında tüm veri seti bir kere karıştırılır**, sonra sırayla sabit boyutlu parçalara (mini-batch'lere) bölünür. Böylece her örnek epoch başına tam bir kez kullanılır, ama sıra her epoch'ta farklı olduğu için rastgelelik faydası korunur. PyTorch'ta `DataLoader(shuffle=True)` tam olarak bunu yapıyor.
- **Stratified / class-balanced batching:** sınıflar dengesiz dağılmışsa (örn. dolandırıcılık tespiti — az sayıda "fraud" örneği, çoğunluk "normal" işlem, roadmap'in kendi hedef alanlarından biri), saf rastgele örneklem her batch'i çoğunluk sınıfla doldurabilir, azınlık sınıfın gradient'e katkısı kaybolur. Bu durumda mini-batch'ler bilerek her sınıftan dengeli oranda örnek içerecek şekilde kurulur.
- **Curriculum learning** (daha niş): örnekleri zorluk sırasına göre (kolaydan zora) sunmak, saf rastgelelik yerine — araştırma aşamasında kullanılan bir teknik, şimdilik bilmen yeterli.
- **Importance sampling** (daha niş): her örneği eşit probability'yle değil, "ne kadar bilgilendirici/yüksek gradient'li" olduğuna göre ağırlıklı seçmek — yine ileri seviye bir teknik.

---

## Bağlantı

Görev 5'teki (`Neuron`/`Layer`/`MLP (Multi-Layer Perceptron)` training loop) toy dataset'in muhtemelen tamamını her adımda kullanacaksın — pratikte bu batch GD'ye denk düşüyor (veri seti zaten küçük, mini-batch'e ihtiyaç yok). Mini-batch/SGD ayrımı, veri gerçekten büyüdüğünde (Faz 3-4, PyTorch, F1 telemetri) anlamlı hale gelecek. Fraud detection ilgi alanınla (roadmap'in kariyer bölümü) kesişen kısım: stratified batching, dengesiz veri setlerinde doğrudan işine yarayacak bir teknik.

**Bağlantılı:** [Hafta 2 — Aktivasyon Fonksiyonları: Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/).
