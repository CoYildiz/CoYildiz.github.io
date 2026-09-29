---
title: "21 · Bigram'ın Sinir Ağı Hali — one-hot ve W"
published: 2026-09-29
description: "Sayım tabanlı bigram ile tek katmanlı sinir ağının aynı şeyi yaptığının gösterimi: one-hot çarpımı aslında satır seçmek."
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

Aynı iş, iki farklı yoldan. Sayım modeli tabloyu **veriden okuyor**, sinir ağı aynı tabloyu **sıfırdan öğreniyor**.

---

## Tek cümlelik fikir

Sayım modelinde elinde hazır bir tablo vardı. Sinir ağında tablo yok — rastgele sayılarla dolu bir `W` var, ve onu düzelte düzelte aynı tabloya benzetiyorsun.

| | Sayım modeli | Sinir ağı |
|---|---|---|
| Tablo nereden geliyor | Veriyi sayarak | Rastgele başlayıp öğrenerek |
| Nasıl güncelleniyor | `N[i,j] += 1` | `W -= lr * W.grad` |
| Olasılığa çevirme | Satırı toplamına böl | `exp` sonra toplamına böl |

---

## 1. Eğitim verisi: `xs` ve `ys`

**Bu bölüm neden var:** Sayım modelinde "eğitim verisi" diye ayrı bir şey yoktu, tabloyu doldurup bitiyordu. Sinir ağına geçerken aynı bigram'ların (girdi, hedef) çiftlerine dönüşmesi gerekiyor.

```
"emma"  →  .e   em   mm   ma   a.

xs (girdi)  :  .    e    m    m    a
ys (hedef)  :  e    m    m    a    .
```

Her bigram bir soru-cevap çifti: **"girdi bu harfse, sonraki harf ne olmalı?"**

Sayım modelinde bu çiftleri sayıyordun. Şimdi model bunları tahmin etmeyi öğrenecek. Veri aynı, kullanım biçimi farklı.

---

## 2. Neden one-hot

**Bu bölüm neden var:** Harfleri 0-26 arası sayı olarak vermek mümkün ama **yanlış** — model `c`'yi `a`'dan üç kat büyük sanır. One-hot bu sahte sıralamayı yok ediyor.

`stoi['e'] = 5`. Bu 5 bir **isim**, miktar değil — sadece "5. kutu" demek.

Ama ağa çiğ olarak `5` verirsen ağ onu miktar sanar. `e` harfini `a`'nın (1) beş katı, `z`'nin (26) beşte biri sayar. Harfler arasında var olmayan bir sıralama ve büyüklük uydurmuş olursun.

One-hot bunu ortadan kaldırıyor. Her harf 27 uzunluğunda bir vektör, sadece kendi konumunda 1:

```
'.'  →  [1, 0, 0, 0, ... 0]
'a'  →  [0, 1, 0, 0, ... 0]
'b'  →  [0, 0, 1, 0, ... 0]
```

Hepsi birbirine eşit uzaklıkta. Hiçbiri diğerinden büyük değil.

---

## 3. `xenc @ W` ne yapıyor — satır seçiyor

**Bu bölüm neden var:** Bu çarpımın bir matrix çarpımı gibi görünüp aslında **indeksleme** yapması, Hafta 4'te embedding'in (`C[X]`) neden doğrudan indekslemeyle yazıldığını açıklıyor. Aynı iş, ucuz hali.

Bir one-hot satırını bir matrixle çarpmak, o matrix'in **bir satırını çekip almak** demek. Başka bir şey olmuyor.

![one-hot ile çarpmak = satır seçmek](/yz50/onehot-secim.png)

`b`'nin one-hot'ında sadece 2. konum 1. Çarpma yapıldığında `W`'nin 2. satırı dışındaki her şey sıfırla çarpılıp yok oluyor. Geriye tek bir satır kalıyor.

**Sayım modelindeki `P[ix]` ile aynı iş.** Orada satırı indeksle alıyordun, burada çarpmayla.

### Peki neden çarpma, indeksleme dururken

Çünkü çarpma bir **sinir ağı layer'ı**. `W` öğrenilebilir bir parametre, `@` derivative'i tanımlı bir işlem — `backward()` zinciri buradan geçebiliyor, `W` güncellenebiliyor.

`N[i,j] += 1` bir gradient zincirine giremez. Sayım modelinin öğrenme diye bir adımı yoktu, tablo veriden doğrudan doluyordu.

> **Not:** Karpathy Part 2'de one-hot'ı bırakıp doğrudan indekslemeye (`C[X]`) geçiyor ve "ikisi aynı şey" diyor. Doğru — one-hot burada **anlatım için**: bunun sıradan bir matrix çarpımı, yani sıradan bir layer olduğunu göstermek için.

---

## 4. Sonraki adım: logits'ten probability'ye

**Bu bölüm neden var:** `W`'nin çıktısı herhangi bir sayı, ama olasılık gerekiyor. `exp` + normalize adımının neden bu sırayla olduğu ve sayım tablosuyla ilişkisi burada.

`xenc @ W`'den çıkan satıra **logits** deniyor. Herhangi bir reel sayı — negatif de olabilir, o yüzden henüz probability değil.

![logits'ten loss'a](/yz50/logits-softmax.png)

İki adımda probability'ye çevriliyor:

1. **`exp`** — hepsi pozitif oluyor. Çıkan sayılar sayım tablosundaki `N`'nin rolünde
2. **Toplamına bölme** — satır toplamı 1 oluyor, artık bir dağılım. Bu da `P`'nin rolünde

İkisine birlikte **softmax** deniyor.

### Buradan çıkan sonuç

`W`, **öğrenilen log-sayımlar**. Sayım modelinde `N`'yi veriden okuyup normalize ediyordun; burada `exp(W)` aynı rolü oynuyor, ve `W` gradient descent ile bulunuyor.

Görev 4'ün "loss'un sayım modelinin loss'una yaklaştığını göster" demesinin sebebi bu: **iki model matematiksel olarak aynı şeye yakınsıyor.** Biri sayarak varıyor, diğeri öğrenerek.

---

## 5. Loss — değişen bir şey yok

**Bu bölüm neden var:** İki modelin aynı loss'a yakınsaması tesadüf değil, ikisinin de aynı şeyi öğrendiğinin kanıtı. Bu bölüm o eşdeğerliği kuruyor.

Gerçek harfin probability'nina bakılıp `-log` alınıyor, sonra ortalaması. Hafta 3'ün başında hesapladığın NLL (Negative Log Likelihood)'in aynısı, bkz. [16](/posts/yz50-16-loss-nll-cross-entropy/).

Tek fark: sayım modelinde loss sadece bir **rapor**du, hesaplayıp bakıyordun. Burada loss **geri besleme** — `backward()` ondan başlayıp `W`'ye kadar gidiyor ve `W`'yi düzeltiyor.

---

## Bütün akış

```
harf 'b'
   ↓  stoi
indeks 2
   ↓  one-hot
[0, 0, 1, 0, ... 0]          şekil (1, 27)
   ↓  @ W                     W şekil (27, 27)
logits                        şekil (1, 27)   ← W'nin 2. satırı
   ↓  exp
pozitif sayılar               ← "sayım" rolünde
   ↓  toplamına böl
probability'ler                   ← "P" rolünde, toplam 1
   ↓  gerçek harfin probability'ninı al, -log
loss
   ↓  backward
W.grad
   ↓  W -= lr * W.grad
biraz düzelmiş W
```

Bu döngü tekrarlandıkça `W`, sayım tablosunun logaritmasına yaklaşıyor.

---

**Bağlantılı:** [18 · PyTorch Broadcasting ve keepdim Tuzağı](/posts/yz50-18-broadcasting/) · [16 · Loss, NLL ve Cross-Entropy](/posts/yz50-16-loss-nll-cross-entropy/) · [24 · Genel Tekrar — micrograd muhasebesi](/posts/yz50-24-tekrar-micrograd-muhasebesi/)
