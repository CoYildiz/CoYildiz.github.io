---
title: "Hafta 3 — `torch.multinomial` Tam Olarak Ne Yapıyor"
published: 2026-09-29
description: "`torch.multinomial` girdisini nasıl yorumluyor, normalize ediyor mu, replacement ne değiştiriyor — belgelenmiş davranışıyla birlikte."
tags:
  - YZ50
  - PyTorch
  - Sampling
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

Sampling'in motoru. Görev 2'de isim üretmek, Görev 4'te modelin çıktısından harf çekmek için kullanılıyor. Önceki notlar: [broadcasting](/posts/yz50-18-broadcasting/) · [sampling döngüsü](/posts/yz50-19-sampling-dongusu/).

---

## Tek cümlede

**Bu bölüm neden var:** Fonksiyonun adı ne yaptığını söylemiyor. Tek cümlelik bir zihinsel model, dokümantasyonu okumaktan daha kalıcı.

Bir probability listesi verirsin, o listeye **uyan** rastgele indeksler döndürür. Değer değil, **indeks** döner.

```python
p = torch.tensor([0.6, 0.3, 0.1])
torch.multinomial(p, num_samples=20, replacement=True)
# tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
```

Çıkan sayılar 0, 1, 2 — yani `p`'nin **konumları**. Uzun vadede yaklaşık %60'ı 0, %30'u 1, %10'u 2 olur.

## İçeride ne oluyor — kümülatif toplam + tek bir rastgele sayı

**Bu bölüm neden var:** Kara kutu olarak kullanmak mümkün ama içi üç satır. Mekanizmayı bilmek, neden tek bir uniform rastgele sayının yettiğini ve sampling'in neden ucuz olduğunu açıklıyor.

Kavramsal olarak standart yöntem (inverse-CDF / ters dağılım örneklemesi):

**1. Kümülatif toplam alınır.** Probabilitylar $[0,1)$ aralığında ardışık dilimlere çevrilir:

```
p       = [0.6,  0.3,  0.1]
cumsum  = [0.6,  0.9,  1.0]

0.0            0.6        0.9   1.0
 ├──────────────┼──────────┼─────┤
 │      0       │    1     │  2  │
 └──────────────┴──────────┴─────┘
   genişlik 0.6   0.3        0.1
```

Dilim genişliği = o indeksin probability'si. Bu yüzden 0'ın dilimi 1'inkinin iki katı.

**2. $[0,1)$ arasından bir rastgele sayı çekilir.**

**3. O sayının düştüğü dilimin numarası döndürülür.**

```
u = 0.42  →  0.42 < 0.6         →  0 döner
u = 0.77  →  0.6 ≤ 0.77 < 0.9   →  1 döner
u = 0.95  →  0.9 ≤ 0.95 < 1.0   →  2 döner
```

Hepsi bu. "Hileli zar" tam olarak böyle çalışıyor: düzgün dağılmış tek bir sayı, eşit olmayan genişlikteki dilimlere düşürülüyor.

![Kümülatif dilimler ve uniform çekiliş](/yz50/multinomial-dilimler.png)

**Doğrulama (2026-09-06, torch 2.14):** aynı fikri `torch.rand` + `cumsum` + `searchsorted` ile elle kurup 200.000 örnek çektim — `multinomial` frekansları $[0.6013, 0.2996, 0.0991]$, elle kurulan $[0.5993, 0.3001, 0.1006]$. Aynı dağılım.

## `replacement` ne yapıyor

**Bu bölüm neden var:** Varsayılanı `False` ve dil modelinde bu **yanlış** — her çağrıda `True` vermen gerekiyor. Sebebini bilmezsen sessiz bir hata kaynağı.

| | ne demek | sonuç |
|---|---|---|
| `True` | çekilen indeks havuzda kalır | aynı indeks tekrar tekrar gelebilir; dağılım hiç değişmez |
| `False` | çekilen indeks havuzdan çıkar | her indeks en fazla bir kez; kalanlar yeniden normalize edilir |

`False` ile `num_samples`, `p`'nin uzunluğunu geçemez:

```
RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
```

**Dil modelinde her zaman `True`.** İki sebep:

1. Teknik: 27 harften çok daha uzun diziler üreteceksin.
2. Kavramsal: `False`, her çekilişten sonra dağılımı değiştirir — bir harf "tükenir". Dil öyle çalışmıyor; `emma`'da iki `m` var. Bir dağılımdan örneklemek demek, dağılımın örnekleme boyunca **sabit kalması** demek.

## Ayrıntılar

- **Girdi toplamı 1 olmak zorunda değil.** `multinomial` weight'leri içeride normalize eder; `[3, 1, 1]` ile `[0.6, 0.2, 0.2]` aynı sonucu verir. Yine de dışarıda açıkça normalize etmek doğru refleks — çünkü Görev 3'te aynı sayıların **log**'unu alacaksın ve orada gerçek probability olması şart.
- **Negatif weight kabul edilmez**, hata verir. Sıfır serbest: o indeks hiç seçilmez.
- **Matrix verilebilir.** $(n, k)$ şeklinde bir tensor verirsen her satırı ayrı bir dağılım sayar ve her satır için ayrı örnek çeker. Bigram tablosunun 27 satırı için tek çağrıda 27 zar atmak mümkün.
- **`generator=g`** verilmezse global rastgelelik akışı kullanılır. Verirsen o üretecin durumu ilerler — bkz. [19](/posts/yz50-19-sampling-dongusu/), aynı seed'in neden videodakiyle aynı sonucu vermediği de orada.
- **Düşük probability ≠ imkânsız.** 0.0065 probabilitylı bir harf ~154 çekilişte bir gelir. Ürettiğin isimlerde tuhaf harfler görmen modelin bozukluğu değil, örneklemenin doğası.

## Neden `argmax` değil

**Bu bölüm neden var:** "En olası harfi seç" sezgisel olarak doğru görünüyor ama üretimi bozuyor. Bu ayrım, sampling ile tahmin arasındaki farkın kendisi.

En yüksek probabilitylı harfi seçseydin model **tek bir isim** üretirdi, hep aynısını: `.`'dan sonra hep en olası harf, ondan sonra hep en olası harf... Döngü ya sabit bir dizide takılır ya da tekrara girer. Çeşitlilik, örneklemenin rastgeleliğinden geliyor — modelin öğrendiği şey "hangi harf" değil, "harflerin **dağılımı**".

---

**İlgili:** [Hafta 3 — Sampling Döngüsü: `P[ix]` ve Seed Tekrarlanabilirliği](/posts/yz50-19-sampling-dongusu/)
