---
title: "11 · Backpropagation Sezgisi"
published: 2026-09-29
description: "Backpropagation'ın sezgisi: her eğitim örneğinin weight'lerden ne istediği, bu isteklerin nasıl toplandığı ve neden 'geriye' gidiyoruz."
tags:
  - YZ50
  - Backpropagation
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

Kaynak: [3blue1brown.com/lessons/backpropagation](https://www.3blue1brown.com/lessons/backpropagation), orijinal video [YouTube'da](https://www.youtube.com/watch?v=Ilg3gGewQ5U). **Bu, YZ50 Hafta 2'nin kendi kaynak listesindeki 3. madde** — resmi, atanmış materyal. Önceki: [05 · Ağ Analizi ve Sınırları](/posts/yz50-05-ag-analizi-ve-sinirlari/). Sonraki: [12 · Backpropagation Calculus](/posts/yz50-12-backpropagation-calculus/) (kesin matematik).

---

## Hızlı Hatırlatma

Ağ: 784 girdi, iki 16-nöronlu gizli layer, 10 çıktı. "Öğrenme" = cost function'ı minimize eden weight/bias'ları bulmak. Tek bir örnek için cost = çıktı ile istenen değer arasındaki farkların karelerinin toplamı; tüm eğitim verisi üzerinden ortalaması, ağın toplam maliyeti.

## Backpropagation Tam Olarak Ne

**Bu bölüm neden var:** Kelime "geriye yayılım" diye çevriliyor ama neyin yayıldığı belirsiz kalıyor. Yayılan şey **hassasiyet** — her parametrenin cost'u ne kadar etkilediği bilgisi.

Negatif gradient, **13.002 boyutlu bir vektör** — her weight/bias'ı ne kadar/hangi yönde dürtmen gerektiğini söylüyor. **Backpropagation, bu negatif gradient'i hesaplama algoritması.**

13.002 boyutta "yön" hayal etmek imkânsız — bunun yerine, gradient'in her bileşeninin **büyüklüğünü**, cost'un o weight'e **ne kadar duyarlı** olduğu olarak oku. Örnek: bir weight'in gradient bileşeni 3.2, başka birininki 0.1 ise — cost, birinci weight'e **32 kat daha duyarlı**. Aynı büyüklükte bir oynatma, birincisinde 32 kat daha fazla cost değişimine yol açar.

## Sezgi — Tek Bir Örnek, Tek Bir Nöron

**Bu bölüm neden var:** Tam formül ([12](/posts/yz50-12-backpropagation-calculus/)) indeksler yüzünden okunmuyor. Önce tek nöronda ne olduğunu görmek, sonra oraya geçmek gerekiyor.

Bir "2" rakamı görüntüsünü düşün. Ağ henüz eğitilmemiş, çıktı layer'ındaki aktivasyonlar rastgele. "2" nöronunun **yukarı**, diğer 9 nöronun **aşağı** gitmesini istiyoruz — nudge'ların büyüklüğü, o nöronun hedeften ne kadar uzak olduğuyla orantılı olmalı (çok yanlışsa büyük dürtme, az yanlışsa küçük).

"2" nöronunun aktivasyonunu artırmak için **3 kaldıraç** var (aktivasyon = önceki layer'ın ağırlıklı toplamı + bias, sonra sigmoid/ReLU'dan geçirilmiş):

1. **Bias'ı artır**
2. **Weight'leri artır**
3. **Önceki layer'ın aktivasyonlarını değiştir** (dolaylı — doğrudan dokunamıyoruz)

## 1. Bias

En basit kaldıraç — etkisi doğrudan ve öngörülebilir. "2" nöronunun bias'ı artırılmalı, diğer 9'unki azaltılmalı (istenen yönle aynı).

## 2. Weight'ler — "En Çok Kâr Getiren" Mantığı

Weight'ler, önceki layer'ın aktivasyonlarıyla **çarpılıyor** — yani zaten **parlak (yüksek aktivasyonlu)** bir nörondan gelen bağlantının weight'inı artırmak, sönük bir nörondan gelene göre **çok daha büyük etki** yapıyor. "En çok kâr" için, weight'leri **kendi aktivasyonlarıyla orantılı** artırmalısın.

**Hebbian teori paraleli:** nörobilimde "birlikte ateşlenen nöronlar birlikte bağlanır" (neurons that fire together, wire together) fikri var — burada da en büyük weight artışları, zaten aktif olan nöronlarla aktif olmasını istediğimiz nöron arasında oluyor. Analoji tam mükemmel değil (ağ gerçekten "2'yi düşünmüyor", etiket bunu zorluyor) ama matematiğin altında bu paralellik gerçek.

## 3. Önceki Layer'ın Aktivasyonları (Dolaylı)

"2" nöronuna **pozitif** weightla bağlı her şey daha parlak, **negatif** weightla bağlı her şey daha sönük olsaydı, "2" nöronu daha aktif olurdu. Yine aynı mantık: istenen değişiklik, **ilgili weightla orantılı.**

Doğrudan aktivasyonlara dokunamayız (sadece weight/bias'a) — ama bunu bir **istek listesi** olarak not ediyoruz. Diğer 9 çıktı nöronunun da (çoğu "azalsın" isteyen) kendi istek listesi var — hepsi **toplanıyor**. Bu toplam istek listesi, bir önceki layer için "istenen nudge"ları veriyor — **işte geri yayılım (backpropagation) tam burada başlıyor:** aynı süreç, bir layer geriye, tekrar tekrar uygulanıyor.

![Backprop wishlist — rekabet eden istekler toplanıyor](/yz50/backprop-wishlist.svg)

## Tüm Eğitim Örnekleri İçin Tekrarlama

**Bu bölüm neden var:** Tek örneğe bakarak öğrenirsen model o örneği ezberler. Ortalama almanın neden zorunlu olduğu burada.

Yukarıdakiler sadece **tek bir "2" örneği** için. Sadece onu dinlersen, ağ her şeyi "2" sınıflandırmaya teşvik edilir! Bu yüzden bu süreç **her eğitim örneği** için tekrarlanır, her biri kendi nudge isteğini kaydeder, sonra **hepsinin ortalaması** alınır — bu ortalama, (yaklaşık olarak) **negatif gradient'in kendisi.**

## Stochastic Gradient Descent — Zaten Derinlemesine Bildiğin Konu

On binlerce örneği her adımda toplamak çok yavaş — pratik çözüm: veriyi karıştır, mini-batch'lere böl (örn. 100'lük), her mini-batch için bir adım at. Tam gradient değil ama iyi bir **yaklaşım**, çok daha hızlı. Benzetme: "yavaş/dikkatli bir adam" yerine "hızlı adımlarla sarhoş gibi sendeleyen bir adam."

*(Bu, [08](/posts/yz50-08-stochastic-gradient-descent/) ve [07](/posts/yz50-07-neden-gradient-descent/)'te zaten çok daha derinlemesine işlediğin konu — burada sade bir tekrar, yeni bilgi yok. Zaten bildiğini görmek iyi bir doğrulama.)*

## Sonuç

Bu sezgisel anlatım, Nielsen'in kitabındaki (zaten okuduğun) kodun mantığını açıklıyor. Kesin matematiği (calculus) için sıradaki ders — [12](/posts/yz50-12-backpropagation-calculus/).

---

## Bağlantı

Bu 3 kaldıraç (bias/weight/prev-activation), senin `Value`/`Neuron`'unun `backward()` metodunun **tam olarak yapması gereken şey** — her `Value` node'u kendi yerel derivative'ini hesaplayıp üstten gelen gradientla çarpıyor, tıpkı burada "her nöronun kendi payına düşen nudge'ı hesaplayıp bir önceki layer'a iletmesi" gibi.

**Bağlantılı:** [05 · Ağ Analizi ve Sınırları](/posts/yz50-05-ag-analizi-ve-sinirlari/) · [12 · Backpropagation Calculus](/posts/yz50-12-backpropagation-calculus/)
