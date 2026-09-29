---
title: "22 · RNN'e Giriş"
published: 2026-09-29
description: "RNN'in tekrarlayan yapısı, gizli durumun taşıdığı bilgi ve 'RNN Turing complete' iddiasının hangi iki koşula bağlı olduğu."
tags:
  - YZ50
  - RNN
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

Sıralı veriyi işlemek için tasarlanmış sinir ağı ailesi. Bu not kavramsal bir referans: ne olduğu, neden var olduğu, hangi sorunu çözdüğü ve nereye evrildiği.

---

## Ne için var

Feedforward ağlar her girdiyi birbirinden bağımsız işler — bir önceki girdiyi hatırlamaz. Sırası önemli olan veride bu yetersiz kalır:

- Bir cümledeki kelimenin anlamı önceki kelimelere bağlıdır
- Bir zaman serisindeki değer bir öncekine bağlıdır
- Bir ses kaydındaki fonem, kendinden öncekilerle birlikte anlam kazanır

RNN bu sıra bağımlılığını yakalamak için tasarlandı.

## Feedforward'dan farkı: geri besleme

**Bu bölüm neden var:** Buraya kadarki bütün ağlarda bilgi tek yönde akıyordu. Tek bir bağlantının geriye dönmesi, modelin ne yapabileceğini kökten değiştiriyor — ve yeni bir sorun sınıfı doğuruyor.

Feedforward'da bilgi tek yönde akar: girdi → gizli layer → çıktı, biter. RNN'de ek olarak bir geri besleme vardır — bir zaman adımındaki çıktı, bir sonraki adımda tekrar girdi olarak ağa verilir.

![Feedforward vs Recurrent](/yz50/rnn-vs-feedforward.png)

## Hidden state — "hafıza"

**Bu bölüm neden var:** "Hafıza" kelimesi mecaz ve yanıltıcı olabiliyor. Gerçekte ne olduğu — her adımda güncellenen sabit boyutlu bir vektör — bunun sınırlarını da açıklıyor.

RNN'in temel yapı taşı, her zaman adımında güncellenen bir **hidden state**. Güncelleme iki şeye bakar:

1. O anki girdi ($x_t$)
2. Bir önceki hidden state ($h_{t-1}$)

Yani ağ "şu ana kadar gördüğüm her şeyin özeti"ni hidden state içinde taşır ve her yeni girdiyle günceller. Feedforward'daki "her girdi bağımsız" varsayımının tam tersi.

### Unrolled görünüm

RNN aslında **tek bir hücre**, ama zaman adımları boyunca tekrar tekrar kullanılır. Görselleştirmenin en yaygın yolu, onu her adım için ayrı bir kutu gibi açarak çizmek:

![RNN unrolled görünümü](/yz50/rnn-unrolled.png)

Her kutu **aynı weight'leri** kullanır — kopyalanmış ağlar değil, aynı hücrenin tekrarı. Soldan sağa akan tek şey hidden state; her adımda o anki girdiyle güncellenir ve bir çıktı ($y_t$) üretilir.

## Nasıl eğitilir

**Backpropagation Through Time (BPTT):** standart backprop'un zaman eksenine genellemesi. Chain rule mantığı aynıdır, sadece "açılmış" (unrolled) graf üzerinde uygulanır. Yeni bir algoritma değil.

**Teacher forcing:** eğitim sırasında modele kendi (muhtemelen hatalı) bir önceki tahminini değil, **doğru** bir önceki değeri vermek. Böylece erken bir hata zincirleme şekilde sonraki tüm tahminleri bozmaz, öğrenme sinyali temiz kalır.

## Temel sorun: vanishing gradient

**Bu bölüm neden var:** [13](/posts/yz50-13-aktivasyon-fonksiyonlari/)'teki aynı sorun, burada derinlik yerine **zaman** boyunca ortaya çıkıyor. Aynı çarpım zinciri, farklı eksen — ve LSTM/GRU'nun varlık sebebi.

Gradient zaman içinde geriye yayılırken çok sayıda adım boyunca üst üste çarpılır. Küçük sayıların tekrar tekrar çarpılması sıfıra çöker.

Sonuç: dizinin başındaki bir bilginin çok sonrasını etkilediği **uzun mesafeli bağımlılıklar** öğrenilemez. Hafıza pratikte birkaç adımda silinir.

## Çözümler: LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit)

| Mimari | Yıl | Ne yapar |
|---|---|---|
| **LSTM** (Long Short-Term Memory) | 1997 | Vanishing gradient'i çözmek için tasarlandı, uzun vadeli bağımlılıklarda standart RNN varyantı oldu |
| **GRU** (Gated Recurrent Unit) | 2014 | LSTM'in daha az parametreli, hesaplama açısından daha verimli alternatifi |

İkisinin de temel fikri **gate** (kapı) mekanizması: bilginin ne kadarının hidden state'te tutulacağına, ne kadarının unutulacağına ağın kendisi karar verir.

## Transformer'lar

**Bu bölüm neden var:** RNN'in sıralı yapısı yalnızca gradient sorunu değil, **paralelleştirilememe** sorunu da getiriyor. Transformer'ın neden kazandığını anlamak için bu ikinci sorunu görmek gerekiyor — YZ50'nin ilerleyen haftalarının konusu.

Recurrence yerine **self-attention** mekanizmasına dayanan mimari. NLP'de baskın hale geldi. İki avantajı:

1. Uzun mesafeli bağımlılıkları RNN/LSTM'den daha iyi yakalar
2. **Paralelleştirilebilir** — RNN'in "önceki adımı bitirmeden sonrakine geçemezsin" zorunluluğu yok, eğitim ciddi biçimde hızlanır

RNN'ler yine de tamamen elden düşmedi: hesaplama verimliliğinin, gerçek zamanlı işlemenin veya verinin doğası gereği sıralı olmasının kritik olduğu yerlerde hâlâ tercih ediliyor.

## Konfigürasyonlar

- **Stacked (Deep) RNN** — birden fazla RNN üst üste; bir layer'ın çıktı dizisi bir sonrakinin girdisi olur. Feedforward'daki "daha fazla layer = daha derin temsil" mantığının karşılığı.
- **Bidirectional RNN** — diziyi hem ileri hem geri yönde işleyen iki ayrı RNN, çıktılar birleştirilir. Bir kelimeyi hem öncesindeki hem sonrasındaki bağlamla anlamak için.
- **Encoder-Decoder (seq2seq)** — iki RNN art arda: encoder girdi dizisini bir özet temsile sıkıştırır, decoder bunu okuyup çıktı dizisini üretir. Makine çevirisinin klasik kurulumu ve **attention mekanizmasının doğrudan öncülü.**

## Kullanım alanları

Ortak nokta: hepsinde veri sıralıdır, bir öğe kendinden öncekilerin bağlamı olmadan tam anlamıyla işlenemez.

- El yazısı tanıma (segment edilmemiş, bitişik yazı)
- Konuşma tanıma
- Doğal dil işleme ve makine çevirisi
- Zaman serisi tahmini
- Zaman serisi anomali tespiti

## Turing-complete olması

**Bu bölüm neden var:** Bu iddia literatürde sıkça, koşulları söylenmeden tekrarlanıyor. Koşullarıyla birlikte bilmek, "RNN her şeyi yapabilir" yanlış anlamasını önlüyor.

RNN'ler teorik olarak Turing complete — **ama iki koşul altında, ve bu koşullar önemsiz değil.**

Sonuç Siegelmann & Sontag'a ait (1992/1995): rasyonel weight'li, lineer-sigmoid hücrelerden kurulu
bir RNN herhangi bir Turing makinesini simüle edebiliyor. İspatın dayandığı iki varsayım:

1. **Keyfi (sınırsız) hassasiyet.** İspatın hilesi tam olarak budur: tek bir nöronun aktivasyonu
   keyfi hassasiyette bir rasyonel sayı olabildiği için, **sonlu sayıda nöron bütün bir Turing
   makinesinin şeridini hafızasında tutabiliyor.** Sayıyı bir "sonsuz uzunlukta kaset" gibi
   kullanıyor
2. **Sınırsız hesaplama süresi** — tekrarlanan adım sayısının girdi uzunluğuyla sınırlanmaması

> **Kritik sonuç:** **sonlu hassasiyetli bir RNN Turing complete değildir** — sonlu sayıda
> durumu olduğu için bir **sonlu durum makinesidir.** Gerçek donanımda float32/float64 ile
> çalıştığın için, eğittiğin hiçbir RNN bu teoremin kapsamına girmiyor.

*(2026-09-20'de kaynaktan doğrulandı. Önceki hali bu iki koşulu yazmıyordu ve teoremi olduğundan
geniş gösteriyordu.)*

Perceptron ağlarının NAND'ı taklit edebildiği için "evrensel" sayılmasıyla aynı aileden bir sonuç,
ve aynı uyarıyı taşıyor: **"teorik olarak yapabilir", "pratikte öğrenir" demek değil.** Burada
üçüncü bir katman daha var: **"teorik olarak yapabilir" bile, gerçek sayı tipleriyle geçerli değil.**

## Kısa tarihçe

```
1980'ler   Elman / Jordan ağları — ilk basit RNN'ler
1997       LSTM — vanishing gradient çözümü
2014       GRU + seq2seq — encoder-decoder makine çevirisi
2017       "Attention is All You Need" — transformer, recurrence'ı
           tamamen self-attention ile değiştirdi
```

---

**Kapsam notu:** Bu not RNN ailesinin genel haritasını çıkarıyor; gate mekanizmalarının matematiği, Hopfield ve echo state ağları, ikinci dereceden ve hiyerarşik varyantlar kapsam dışı bırakıldı.

**Bağlantılı:** [13 · Sigmoid vs Tanh vs ReLU](/posts/yz50-13-aktivasyon-fonksiyonlari/)
