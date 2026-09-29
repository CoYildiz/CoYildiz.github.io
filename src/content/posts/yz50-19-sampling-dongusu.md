---
title: "19 · Sampling Döngüsü"
published: 2026-09-29
description: "Karakter üreten sampling döngüsünün her satırı: olasılık dağılımından çekim, seed ile tekrarlanabilirlik ve sık yapılan indeks hataları."
tags:
  - YZ50
  - Dil Modeli
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

Görev 2'nin ikinci yarısı: probability matrix'inden isim üretmek. İki ayrı kafa karışıklığını kapatıyor — `P[ix]`'in ne döndürdüğü, ve aynı seed'in neden videodakiyle aynı çıktıyı vermediği. Önceki not: [18 · PyTorch Broadcasting ve keepdim Tuzağı](/posts/yz50-18-broadcasting/).

---

## `P[ix]` bir şeyi güncellemiyor, sadece okuyor

**Bu bölüm neden var:** Döngüdeki `ix` değişkeni hem okunuyor hem yazılıyor ve bu, "tablo güncelleniyor" yanılsaması yaratıyor. Tablo sabit; değişen tek şey hangi satıra baktığın.

`P` sabit bir tablo — $(27, 27)$, hesaplandıktan sonra hiç değişmiyor.

```python
P.shape       # torch.Size([27, 27])   ← tüm tablo
P[0].shape    # torch.Size([27])       ← tek bir satır
```

`P[ix]` = "tablonun `ix`. satırını ver". Dönen şey 27 elemanlı bir vektör: **`ix` harfinden sonra hangi harfin gelme probability'si ne.** Toplamı 1, çünkü `keepdim=True` ile satır bazında normalize edildi.

Döngüde değişen `P` değil, **`ix`**. Her turda farklı bir satır okunuyor:

| `ix` | `P[ix]` neyin dağılımı |
|---|---|
| 0 | `.` sonrası — yani kelimelerin **ilk harfi** |
| 1 | `a` sonrası |
| 3 | `c` sonrası |
| ... | ... |

Yani `P`'nin her satırı ayrı bir zar. `ix` hangi zarı atacağını seçiyor.

## Döngünün tam akışı

**Bu bölüm neden var:** Üretim döngüsü dört satır ama dördü de birbirine bağlı. Sırayı bir kez açıkça yazmak, sonraki modellerde (MLP, transformer) aynı döngünün tanınmasını sağlıyor — sampling mantığı değişmiyor, yalnızca olasılığı kim üretiyor o değişiyor.

`ix = 0` ile başlanır (`.` = kelime başlangıcı). Her tur: satırı oku → zar at → çıkan indeksi hem çıktıya ekle hem bir sonraki satır numarası yap. `ix` tekrar 0 olunca (`.` çekilince) kelime biter.

Gerçek bir üretim, adım adım (seed 2147483647, torch 2.14):

```
ix= 0 (.)  →  P[0]  = '.' sonrası    →  3 (c)   probability 0.0481
ix= 3 (c)  →  P[3]  = 'c' sonrası    →  5 (e)   probability 0.1560
ix= 5 (e)  →  P[5]  = 'e' sonrası    → 24 (x)   probability 0.0065
ix=24 (x)  →  P[24] = 'x' sonrası    → 26 (z)   probability 0.0273
ix=26 (z)  →  P[26] = 'z' sonrası    →  5 (e)   probability 0.1555
ix= 5 (e)  →  P[5]  = 'e' sonrası    →  0 (.)   probability 0.1950
sonuç: cexze.
```

Dikkat: 3. adımda probability'si 0.0065 olan `x` seçildi. Düşük probability **imkânsız değil** — 154 denemede bir civarı. Çıkan isimlerin tuhaflığının bir kısmı buradan geliyor, model bozuk olduğundan değil.

Ayrıca `e` iki kez geçti ve iki seferinde de **aynı** `P[5]` satırı kullanıldı — ama farklı sonuçlar çıktı (`x`, sonra `.`). Zar aynı, atış farklı.

![Sampling döngüsünün 27x27 matrix üzerindeki yürüyüşü](/yz50/bigram-sampling.gif)

## Neden Karpathy ile aynı çıktı gelmiyor

**Bu bölüm neden var:** Aynı kodu yazıp farklı isimler görmek "bir şeyi yanlış yaptım" hissi veriyor. Sebebi kod değil, rastgelelik durumu — ve bunu bilmek gereksiz hata aramayı önlüyor.

Aynı seed, aynı kod, farklı sonuç — ve bu bir hata değil.

Seed sadece **aynı PyTorch sürümünde** aynı sayı akışını garanti ediyor. PyTorch'un kendi Reproducibility notu bunu açıkça söylüyor: *"reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms"* ([PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html)). Karpathy videoyu torch 1.x ile çekti; buradaki kurulum 2.14. `multinomial`'ın rastgele sayı tüketme biçimi arada değişmiş olabilir, o zaman aynı seed'den farklı bir dizi çıkar.

**Modelin doğruluğu buradan anlaşılmaz.** Doğrulama, çıktı stringlerini videoyla karşılaştırmak değil, sayısal sağlamalar:

- `N.sum() == 228146` (toplam bigram sayısı)
- `N[0].sum() == 32033` (kelime sayısı — her kelime bir başlangıç bigram'ı üretir)
- `P.sum(1)` hepsi 1, `P.sum(0)` **değil** (satır bazında normalize edildiğinin kanıtı)
- Görev 3'teki NLL (Negative Log Likelihood) değeri — asıl ölçüt bu

Bu dördü tutuyorsa model doğrudur, ürettiği isimler videodakinden farklı olsa bile.

## Neden `g` döngüden önce yeniden tohumlanıyor

```python
g = torch.Generator().manual_seed(2147483647)
```

Bir generator kullanıldıkça durumu ilerler. Yukarıdaki oyuncak `torch.rand(3, generator=g)` denemesi de bu akıştan tüketir. Döngüden hemen önce yeniden tohumlamak, akışı başa sarar — böylece üstteki deneme satırlarını silsen de silmesen de aynı isimler çıkar.

---

**İlgili:** [18 · PyTorch Broadcasting ve keepdim Tuzağı](/posts/yz50-18-broadcasting/)
