---
title: "MNIST'i Görselleştirme — Boyut İndirgeme Teknikleri (Kısa Not)"
published: 2026-09-29
description: "784 boyutlu MNIST'i iki boyutta görmek: PCA, t-SNE ve UMAP'in ne gösterdiği, neyi çarpıttığı."
tags:
  - YZ50
  - Görselleştirme
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

Kaynak: [colah's blog — Visualizing MNIST](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) (kullanıcı yapıştırdı).

**⚠️ Durum: bilinçli olarak kısa/yüzeysel tutuldu, derinlemesine çalışılmadı.** Bu roadmap'te henüz hiçbir fazda planlı bir konu değil (ne Hafta 1 ne Hafta 2, Faz 3.0'da PCA'ya bir cümlelik değinme dışında) — Faz 3'e gelince "yapıldı" sayılmaz. YZ50 Hafta 2 deadline'ı 2 gün kaldığı için detaylı işlemek şu an doğru zamanlama değil, sadece kaynağı kaybetmemek için not düşülüyor.

## Ana Fikir

**Bu bölüm neden var:** 784 boyutlu veriye "bakmak" mümkün değil, ama modelin ne öğrendiğini
anlamak için bir şekilde bakmak gerekiyor. Bu not, o problemi çözmeye çalışan yöntem ailesinin
haritası — henüz çalışılmadı, kaynağı kaybetmemek için duruyor.

784 boyutlu (28×28 piksel — her piksel bir eksen) MNIST verisini insan gözünün görebileceği 2-3 boyuta indirgeyip görselleştirme denemeleri:

![784 boyuttan 2 boyuta indirgeme (temsili)](/yz50/mnist-boyut-indirgeme.png)

*(Soldaki gerçek MNIST pikselleri değil, sadece "yüksek boyutta göze hiçbir yapı görünmüyor" fikrini temsil eden rastgele veri; sağdaki de gerçek bir t-SNE çıktısı değil, "boyut indirgendiğinde sınıflar kümeleşir" fikrinin temsili bir illüstrasyonu.)*

Karşılaştırılan yöntemler:

- **PCA (Principal Component Analysis):** verinin en çok yayıldığı yönleri (eksenleri) bulur, doğrusal bir izdüşüm — hızlı ama MNIST gibi nonlineer yapıları iyi ayıramıyor.
- **MDS (Multidimensional Scaling):** noktalar arası orijinal mesafeleri korumaya çalışan bir optimizasyon (gradient descent ile) — "yaylarla bağlı noktalar" fiziksel benzetmesi.
- **Sammon's Mapping:** MDS'in yakın noktalara daha çok önem veren versiyonu.
- **Graph-based (nearest-neighbor graph):** her noktayı en yakın komşularına bağlayıp graf çizim algoritmasıyla yerleştirme.
- **t-SNE:** komşuluk yapısını (topolojiyi) korumaya odaklanan, derin öğrenme camiasında en popüler teknik — kümeleri/alt-kümeleri ortaya çıkarmakta çok başarılı ama local minimum'a takılabiliyor.

## Nereye bağlanıyor

**Bu bölüm neden var:** Bu tekniklerin ne zaman gerçekten gerekeceğini işaretlemek için — şimdi
değil, ama "hiç" de değil.

PCA, birkaç gün önce konuştuğumuz "unsupervised learning roadmap'te var mı" sorusunun cevaplarından biri — Faz 3.0'da "PCA'nın temeli" olarak zaten bir cümleyle geçiyor. Bu post, o kısma gelince (Faz 3, aylar sonra) iyi bir ek kaynak olabilir. **Şimdi değil.**

