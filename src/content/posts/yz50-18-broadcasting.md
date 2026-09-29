---
title: "Hafta 3 — PyTorch Broadcasting ve `keepdim` Tuzağı"
published: 2026-09-29
description: "PyTorch broadcasting kurallarının adım adım işleyişi ve `keepdim=True` unutulduğunda sessizce yanlış sonuç veren klasik tuzak."
tags:
  - YZ50
  - PyTorch
  - Broadcasting
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

Kaynak: [PyTorch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) (tek sayfa, resmî referans). YZ50 Hafta 3'ün kendi kaynak listesindeki 3. madde. Görev 2'de (sayım tablosunu probability'ye çevirme) doğrudan lazım.

---

## Problem ne

**Bu bölüm neden var:** Broadcasting hata vermiyor — yanlış kullandığında kod çalışıyor ve **sessizce yanlış model** üretiyor. Bu notun varlık sebebi bu: yakalanması zor bir hata sınıfı.

Farklı şekilli iki tensor'u toplarken/çarparken/bölerken NumPy ve PyTorch, küçük olanı büyüğün şekline **kopyalamadan uzatır**. Buna broadcasting deniyor. Rahatlık için var ama iki tarafı keskin: şekiller "tesadüfen" uyduğunda hata vermez, sessizce yanlış hesap yapar.

## Kural — sağdan hizala

**Bu bölüm neden var:** Broadcasting'in ne zaman devreye girdiği ezberlenebilir bir kurala dayanıyor. Kuralı bilmezsen hangi işlemin yayılacağını tahmin etmek zorunda kalıyorsun.

İki tensor'un şekilleri **sağdan sola** hizalanır. Her eksende üç durumdan biri olmalı:

1. Boyutlar eşit, veya
2. Biri 1 (o eksende tekrarlanır), veya
3. O eksen bir tarafta hiç yok (eksik eksenler **sola** 1 olarak eklenir).

Üçü de tutmuyorsa `RuntimeError`.

```python
A = torch.rand(3, 4)
b = torch.rand(4)

# hizalama:
#   A: (3, 4)
#   b:    (4)  →  (1, 4)     ← eksik eksen SOLA eklendi
# sonuç: (3, 4)              ← b'nin tek satırı 3 kez okunur
(A + b).shape   # torch.Size([3, 4])
```

Buradaki asıl bilgi: `b` **satır** gibi davrandı, sütun gibi değil. Uzunluğu 3 olan bir vektörle aynı şeyi denersen patlar, çünkü sağdan hizalama 3 ile 4'ü karşılaştırır:

```python
c = torch.rand(3)
A + c    # RuntimeError: size 3 vs 4 at dimension 1
```

Sütun yönünde yaymak istiyorsan eksenini kendin vermek zorundasın: $(3,1)$ ile $(3,4)$ uyumlu, $(3,)$ ile değil.

## `dim` argümanı: 0 ve 1 neyi temsil ediyor

**Bu bölüm neden var:** `dim=0` ile `dim=1` karıştırıldığında ortaya çıkan sonuç yine bir tensor — hata yok, sadece yanlış eksen. Hangisinin ne demek olduğunu bir kez oturtmak gerekiyor.

`0` ve `1`, tensörün **eksenleri**. Bir matrix'te eksen 0 satır yönü, eksen 1 sütun yönü.

En sağlam ezber: **verdiğin sayı, yok olan eksendir.**

```
M = 1  2  3
    4  5  6
    7  8  9

M.sum(0) = [12, 15, 18]    0. eksen yok oldu → SÜTUN toplamları
                            (1+4+7, 2+5+8, 3+6+9)

M.sum(1) = [6, 15, 24]     1. eksen yok oldu → SATIR toplamları
                            (1+2+3, 4+5+6, 7+8+9)
```

![dim=0 ve dim=1 hangi yönde topluyor](/yz50/eksen-sum-dim.png)

Kafa karıştıran nokta: `sum(0)` dediğinde satırları topluyorsun ama sonuç **sütun** toplamları oluyor. Topladığın şey ile geriye kalan şey farklı. Doğru okuma "0'ı topla" değil, **"0'ı yok et"**.

### Bigram tablosunda hangisi doğru

`P`'nin her satırı "şu harften sonra ne gelir" dağılımı. Bir satırın toplamının 1 olması isteniyor, yani her satır **kendi toplamına** bölünmeli — satır toplamı `sum(1)` ile geliyor.

`sum(0)` yazılsaydı sütun toplamlarına bölünürdü: "a'dan sonra ne gelir" yerine "a'dan önce ne gelirdi" tarafı normalize edilmiş olurdu. Model sessizce anlamsızlaşırdı.

## `sum(dim)` neden boyut düşürür

**Bu bölüm neden var:** `keepdim` tuzağının kökü burada. Boyutun neden düştüğünü anlamadan `keepdim=True`'yu ezberlemek, bir sonraki farklı şekilli tensorda yine yanılmak demek.

`dim` argümanı "hangi ekseni yok et" demek. $(27, 27)$ bir matrix'te:

```python
M.sum(1).shape                  # torch.Size([27])     ← 1. eksen yok oldu
M.sum(1, keepdim=True).shape    # torch.Size([27, 1])  ← 1. eksen 1 olarak kaldı
```

İkisi de aynı 27 sayıyı içeriyor — **satır toplamları**. Fark sadece şekilde. Ve tam olarak bu şekil farkı, bir sonraki işlemin doğru mu yanlış mı olduğunu belirliyor.

## Tuzağın tam mekaniği

**Bu bölüm neden var:** Yukarıdaki iki kuralın birleşiminden hata nasıl doğuyor — adım adım. Bu bölüm olmadan "keepdim kullan" bir batıl inanç.

Amaç: her satırı kendi toplamına bölmek, yani her satır bir probability dağılımı olsun.

**Yanlış yol** — `keepdim` yok:

```
M         (27, 27)
M.sum(1)      (27)  →  (1, 27)   ← sola eklendi, SATIR oldu
```

Bölme yapılır, hata alınmaz — çünkü 27 ile 27 uyuşur. Ama yayılma yönü yanlış: her satır, **sütun toplamlarına** bölünmüş olur. Aslında `M[i,j] / M.sum(1)[j]` hesaplanıyor; oysa istediğin `M[i,j] / M.sum(1)[i]` idi.

**Doğru yol** — `keepdim=True`:

```
M                        (27, 27)
M.sum(1, keepdim=True)   (27, 1)   ← 1 olan eksen genişler, SÜTUN kalır
```

Şimdi $i$. satırın her elemanı $i$. satırın toplamına bölünüyor.

## Görselleştirme — 3×3'lük elle takip edilebilir örnek

27×27 gözle izlenemiyor, aynı şey 3×3'te birebir aynı mekanizmayla oluyor.

```
M =  3  1  0        satır toplamları:  4
     0  2  2                           4
     1  1  6                           8
```

### Yanlış: `M / M.sum(1)`

`M.sum(1)` şekli $(3,)$. Sağdan hizalama, eksik ekseni **sola** ekler → $(1,3)$. Yani vektör bir **satır** olarak yorumlanır ve aşağı doğru çoğaltılır:

```
        payda böyle yayılır
        ┌─────────────┐
M   =   3  1  0       │   4  4  8   ← aynı satır
        0  2  2       │   4  4  8   ← aynı satır
        1  1  6       │   4  4  8   ← aynı satır
                      └─────────────┘

sonuç = 3/4  1/4  0/8  =  0.75  0.25  0.00     satır toplamı 1.00
        0/4  2/4  2/8     0.00  0.50  0.25     satır toplamı 0.75  ✗
        1/4  1/4  6/8     0.25  0.25  0.75     satır toplamı 1.25  ✗
```

Her sütun, **kendi indeksindeki satırın** toplamına bölündü. 3. sütun 8'e bölündü çünkü 3. satırın toplamı 8 — oysa 3. sütunla 3. satırın hiçbir ilgisi yok. Satır toplamları 1 çıkmıyor, yani elde geçerli bir probability dağılımı yok.

**Ve PyTorch hata vermiyor**, çünkü 3 ile 3 uyuşuyor.

### Doğru: `M / M.sum(1, keepdim=True)`

Şekil $(3,1)$. Ekseni koruduğun için vektör bir **sütun**; 1 olan eksen sağa doğru genişler:

```
        payda böyle yayılır
        ┌─────────────┐
M   =   3  1  0       │   4  4  4   ← satırın kendi toplamı
        0  2  2       │   4  4  4
        1  1  6       │   8  8  8
                      └─────────────┘

sonuç = 3/4  1/4  0/4  =  0.750  0.250  0.000    satır toplamı 1.0  ✓
        0/4  2/4  2/4     0.000  0.500  0.500    satır toplamı 1.0  ✓
        1/8  1/8  6/8     0.125  0.125  0.750    satır toplamı 1.0  ✓
```

Sütun toplamları $[0.875, 0.875, 1.25]$ — 1 değil, olması da gerekmiyor. Her **satır** bir dağılım; sütunlar dağılım değil.

![Yanlış ve doğru normalizasyon yan yana](/yz50/broadcasting-keepdim.png)
### Özet: tek fark yayılma yönü

| | şekil | yayılma yönü | her eleman neye bölünür |
|---|---|---|---|
| `M.sum(1)` | $(3,)\to(1,3)$ | aşağı ↓ | sütun indeksine karşılık gelen satırın toplamı ✗ |
| `M.sum(1, keepdim=True)` | $(3,1)$ | sağa → | kendi satırının toplamı ✓ |

## "Kopyalama" gerçekte olmuyor

Yukarıdaki şemalarda paydayı tekrarlanmış olarak çizdim, anlaşılsın diye. PyTorch bunu fiziksel olarak yapmıyor: uzatılan eksende **stride 0** kullanılıyor, yani bellekte tek bir kopya var ve aynı adres tekrar tekrar okunuyor. 27×27 için 27 katlık bir bellek maliyeti yok.

Pratik sonucu: yayılmayı elle `repeat` ile taklit etmeye çalışmak hem gereksiz hem pahalı. Doğru refleks, veriyi kopyalamak değil **şekli düzeltmek** — `keepdim=True` tam olarak bu.

## Neden 27×27 özellikle tehlikeli

**Bu bölüm neden var:** Kare matrix'te yanlış eksende toplamak **hata vermiyor**, çünkü boyutlar tesadüfen uyuyor. Bigram tablosunun tam olarak 27×27 olması bu tuzağı görünmez kılıyor.

Kare matrix olduğu için. $(27, 50)$ bir tabloda `keepdim` unutulsaydı 50 ile 27 çakışır, PyTorch hata verirdi ve saniyesinde anlardın. Karede iki boyut tesadüfen eşit olduğu için hata mekanizması devreye girmiyor — kod çalışır, sayılar makul görünür, model yanlış olur. **Sessiz hata, gürültülü hatadan pahalıdır.**

## Nasıl yakalanır

**Bu bölüm neden var:** Hata sessiz olduğu için kendi kontrolünü yazmak zorundasın. Hangi tek satırlık kontrolün bunu yakaladığı burada.

Ayrı bir doğrulama gerekiyor, çünkü çalışma/çalışmama testi bunu ayırt etmiyor:

- Bölmeden önce iki tarafın `.shape`'ini yazdır. `(27, 27)` ve `(27, 1)` görüyorsan doğru, `(27, 27)` ve `(27)` görüyorsan yanlış.
- Bölmeden sonra **satır toplamlarının 1 olduğunu** kontrol et. Yanlış yaptıysan bunun yerine sütun toplamları 1 çıkar — ikisi aynı anda 1 olmaz (matrix simetrik değilse).
- Sampling sonuçları gözle makul görünebilir, bu bir kanıt değil. Yanlış normalize edilmiş bir tablo da harf üretir, sadece yanlış dağılımdan üretir.

## Bellek notu

Broadcasting fiziksel kopya üretmez. Uzatılan eksende **stride 0** kullanılır, yani aynı bellek bölgesi tekrar tekrar okunur. Bu yüzden $(27,1)$ bir vektörü $(27,27)$'ye "genişletmek" bedava sayılır — 27 kat bellek harcanmıyor. `expand` ile `repeat` arasındaki fark da tam olarak bu: `expand` stride oyunu, `repeat` gerçek kopya.

## Genel kural olarak akılda kalacak şey

> Bir eksende indirgeme yapıp (`sum`, `mean`, `max`) sonucu **aynı tensor'la** tekrar işleme sokacaksan, `keepdim=True` neredeyse her zaman doğru olandır.

İndirgenmiş sonucu başka bir yere yazdıracaksan veya tek başına kullanacaksan `keepdim`'e gerek yok.

## micrograd'la bağlantı

[Hafta 2](/posts/yz50-12-backpropagation-calculus/)'de `Value` skalerlerle çalışıyordu — her düğüm tek bir sayı, broadcasting diye bir kavram yok. PyTorch'ta aynı chain rule tensorlar üzerinde işliyor ve **şekil hatası gradient'i de sessizce bozuyor**: yanlış yönde yayılmış bir bölme, `backward()` çağrıldığında da yanlış derivative'leri geri akıtır. Yani bu bir "veri hazırlama detayı" değil, doğrudan öğrenmenin doğruluğu meselesi.

---

**İlgili:** [Hafta 2 — Backpropagation Calculus (3Blue1Brown, kesin matematik)](/posts/yz50-12-backpropagation-calculus/)
