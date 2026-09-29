---
title: "Hafta 1 / 3Blue1Brown Ara Bölüm — Analyzing Our Neural Network"
published: 2026-09-29
description: "Eğitilmiş ağın weight'lerine bakınca ne görüyoruz: 'kenar dedektörü' hikâyesinin nerede tutmadığı ve rastgele gürültünün neden yüksek güvenle sınıflandırıldığı."
tags:
  - YZ50
  - Sinir Ağları
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

Kaynak: [3blue1brown.com/lessons/neural-network-analysis](https://www.3blue1brown.com/lessons/neural-network-analysis), orijinal video [YouTube'da](https://www.youtube.com/watch?v=Ilg3gGewQ5U) değil — bu, gradient descent (Video 2) ile backpropagation (Video 4, "What is backpropagation really doing?") arasına sıkıştırılmış bir **ara bölüm** (interlude), 3Blue1Brown'ın kendi sitesinde "Chapter 3" olarak numaralanmış. Önceki: [Video 2 notları](/posts/yz50-06-gradient-descent/). **Not:** Bu, YZ50'nin kendi "Video 3"ü (Karpathy'nin numerical derivative videosu) ile karıştırılmamalı — farklı bir kaynak, farklı numaralandırma.

---

## Neden Bu Bölüm Var

Backpropagation'ı (nasıl eğitildiğini) anlatmadan önce, eğitimin **sonucunu** görmek için bir mola. Amaç: eğitilmiş ağı analiz edip ne yaptığını ve neden öyle davrandığını görmek.

## Ağın Performansı

**Bu bölüm neden var:** Bir sayının (%96) iyi mi kötü mü olduğu, neyle kıyaslandığına bağlı. Ölçüt olmadan bu sayı hiçbir şey söylemiyor.

İki gizli layer, her biri 16 nöron (seçimin sebebi çoğunlukla **estetik**, prensipli bir gerekçe yok). Ham haliyle ~%96 doğruluk. Layer yapısıyla oynayınca ~%98'e çıkarılabiliyor. Modern (daha sofistike) ağlar %99.75'e kadar çıkabiliyor — [MNIST veri tabanına](http://yann.lecun.com/exdb/mnist/) bakınca, çoğu insanın bile bunu geçebileceğinden emin değil yazar.

## Beklenti vs Gerçek — Asıl Can Alıcı Bulgu

**Bu bölüm neden var:** [04](/posts/yz50-04-sinir-agi-nedir/)'te layer'ların "kenar bulur, sonra şekil bulur" diye çalışacağı **umudu** kurulmuştu. Bu bölüm o umudun gerçekleşmediğini gösteriyor — ve yüksek doğruluğun anlamlı temsil öğrenmekle aynı şey olmadığını.

Video 1'de bir **tahmin/umut** vardı: ikinci layer'daki bir nöronun weight'leri, görselleştirildiğinde tanınabilir bir piksel deseni (kenar/loop gibi) gösterecekti. **Gerçekte hiç öyle değil** — weight'ler neredeyse **rastgele** görünüyor, gevşek bazı örüntüler var ama beklenen "kenar dedektörleri" değil.

**Yorum:** "13.002 boyutlu, kavranamayacak kadar büyük" weight/bias uzayında, ağ çoğu görüntüyü doğru sınıflandıran ama **genellenebilir, yorumlanabilir örüntüler öğrenmemiş** bir "mutlu küçük local minimum" bulmuş. **Yüksek doğruluk ≠ modelin "doğru" ya da anlamlı temsiller öğrendiği** — sadece eğitim dağılımı için işe yarayan BİR noktaya düşmüş, o kadar.

![Gizli layer weight'leri dedektör gibi görünmüyor](/yz50/agirliklar-dedektor-degil.png)

**Bu iddia burada ölçüldü** (2026-09-20): 784-30-10 bir MLP (Multi-Layer Perceptron) MNIST üzerinde eğitildi, test
doğruluğu **%96.31** — yani videodaki ağla aynı ligde. Yukarıdaki 12 kare, gizli layer'ın ilk 12
nöronunun weight'lerinın `28×28` resim hâli. Kırmızı pozitif, mavi negatif weight.

Gevşek bazı yapılar seçilebiliyor (merkezde yoğunlaşma, kenarlarda sönme) ama **hiçbiri temiz bir
kenar ya da ilmek dedektörü değil.** Not: bu ağ %96 doğru çalışıyor. Yani bu görüntü
"model bozuk" demiyor — "çalışan bir model böyle görünüyor" diyor.

## Rastgele Gürültü Deneyi — Çarpıcı Bir Başarısızlık Modu

**Bu bölüm neden var:** Model yalnızca yanılmıyor, **emin olarak** yanılıyor. "Bilmiyorum" diyememesi bir eksiklik değil, hiç istenmemiş olmasının sonucu — ve production ML'de kalibrasyon başlığının çıkış noktası.

Ağa tamamen rastgele piksel gürültüsü (hiçbir rakam olmayan bir şey) verildiğinde, ağ bunun bir **"5" olduğunu %yüksek güvenle** iddia ediyor — gerçek bir 5 gördüğünde ne kadar eminse, bu saçma girdide de o kadar emin. Ağın **belirsizlik** (uncertainty) kavramı yok — hiçbir zaman "bilmiyorum" demeyi öğrenmedi, çünkü hiçbir zaman ondan bu istenmedi. Cost function'ı sadece "10 rakamdan hangisi" sorusuna cevap arıyordu, "bu gerçekten bir rakam mı" sorusuna hiç değil.

**Sebep:** eğitim seti çok dar bir "evren" — hep ortalanmış, sabit boyutlu, hareketsiz rakamlar. Modelin cost function'ı hiçbir zaman "kararsız olmak" için bir teşvik içermedi.

![Rastgele gürültüye yüksek güvenle rakam adı veriyor](/yz50/gurultuye-yuksek-guven.png)

**Aynı modelle ölçüldü:** 2000 kare saf rastgele gürültü verildi. Ortalama güven **%67**,
örneklerin **%11'i %90 üstü güvenle** sınıflandırıldı — hiçbiri rakam değilken.

Beklenmedik bir ayrıntı: soldaki dört gürültü karesinin **dördü de "8"** olarak etiketlendi. Ağın
tanımadığı şeyler için sessiz bir "varsayılan sınıfı" var; bu bir karar değil, weight'lerin
rastgele girdiyle çarpımının nereye düştüğünün sonucu.

Histogramda yeşil (gerçek rakamlar) `1.0`'a yapışmış — beklenen davranış. Kırmızının (gürültü)
`0.4` ile `1.0` arasına yayılması sorun: **modelin "bilmiyorum" diyebileceği bir yer yok**, softmax
çıktıyı her hâlükârda 10 sınıfa bölüştürmek zorunda.

*(Bu, senin roadmap'indeki fraud detection/production ML ilgi alanınla doğrudan kesişiyor — "confidently wrong" [güvenle yanlış] gerçek sistemlerde ciddi bir sorun, calibration/out-of-distribution detection gibi konuların çıkış noktası tam burası.)*

## Neden Yine de Değerli — "Eski Teknoloji, Gerekli Başlangıç Noktası"

**Bu bölüm neden var:** Önceki iki bölüm ağı kötülüyor gibi okunabilir. Bu mimarinin neden hâlâ öğrenilmesi gerektiğini söylemeden bırakmak yanıltıcı olur.

Bu ağın hayal kırıklığı yaratan yönleri, onu değersiz kılmıyor — bu 1980-90'ların fikri, daha modern varyantlara geçmeden önce **anlaşılması gereken bir temel.** Yine de gerçek problemleri çözebiliyor, ama derinlemesine baktıkça "zeki" görünmekten uzaklaşıyor.

## Kırılganlık — Kaydırma/Ölçek Değişimine Duyarlılık

**Bu bölüm neden var:** Modelin en somut sınırı bu ve sebebi mimaride: piksellerin komşuluk bilgisi hiç kullanılmıyor. CNN'in varlık sebebini erkenden vermek için burada.

Ağ, sadece **ortalanmış ve doğru boyuttaki** görüntülerle eğitildi. Rakam çok büyük/küçük ya da merkezden kaymışsa ağ şaşırıyor. **Sebep:** eğitim algoritmasının, bir bölgede öğrenilen bir örüntünün başka bir bölgeye **transfer edilebileceğine** dair hiçbir mekanizması yok — hatta hangi pikselin hangisine komşu olduğu bilgisini bile hiç kullanmıyor (her piksel bağımsız bir girdi gibi davranılıyor).

**İleri bağlantı:** bu tam olarak **convolutional neural network (CNN)**'lerin çözdüğü problem — roadmap'in Faz 4.1'inde ("CNN'e giriş") zaten planlı, bu bölüm o motivasyonu erkenden veriyor.

![Birkaç piksel kaydırma doğruluğu çökertiyor](/yz50/kaydirma-kirilganligi.png)

**Ölçüm:** test setinin tamamı yatay olarak kaydırıldı, başka hiçbir şey değişmedi.

| Kaydırma | 0 px | 1 px | 2 px | 3 px | 4 px | 6 px |
|---|---|---|---|---|---|---|
| Doğruluk | %96.3 | %92.5 | %74.8 | %56.1 | %34.1 | %9.9 |

**İki piksel** doğruluğu %96'dan %75'e düşürüyor. **Altı piksel** modeli rastgele tahmin
seviyesine (%10) indiriyor — insan gözü için hâlâ apaçık okunabilen bir rakamda.

Sebep notun yukarıda yazdığı şey: model hangi pikselin hangisine komşu olduğunu **bilmiyor**.
Kaydırılmış bir 7, onun için tamamen başka bir 784 boyutlu vektör. CNN'in çözdüğü problem tam
olarak bu.

*Görseller ve ölçümler [img/make_network_limits_figures.py](/yz50/make_network_limits_figures.py)
ile üretiliyor — script modeli her koşuda yeniden eğitiyor, sayılar tohuma bağlı olarak birkaç
ondalık oynayabilir.*

## Sıradaki Konu

Bölüm, "backpropagation, ağ eğitiminin asıl beygirgücü" diyerek bir sonraki derse (3Blue1Brown'ın kendi "What is backpropagation really doing?" videosu) yönlendiriyor — bu, **YZ50 Hafta 2'nin kendi kaynak listesindeki 3. madde**, yani bu ara bölümü bitirmen seni gerçekten resmi Hafta 2 materyaline bir adım daha yaklaştırıyor (Karpathy'nin videosuyla birlikte, alternatif/tamamlayıcı bir anlatım olarak).

---

## Bağlantı

**Bağlantılı:** [Video 1 notları](/posts/yz50-04-sinir-agi-nedir/) · [Video 2 notları](/posts/yz50-06-gradient-descent/)
