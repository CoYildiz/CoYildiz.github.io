---
title: "Hafta 2 — CLT (Central Limit Theorem — Merkezi Limit Teoremi)'nin Gerçek İspatı + Hangi İstatistik/Matematik Derslerine Bakmalı"
published: 2026-09-29
description: "CLT'nin ispatının ana hattı (karakteristik fonksiyon), gradient ortalamalarıyla ilişkisi ve ML için hangi istatistik derslerine bakmak gerektiği."
tags:
  - YZ50
  - İstatistik
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

Kaynak: kullanıcı sorusu (2026-08-28) — "anket analojisi diyip geçme, matematiğini göster" + "hangi istatistik/matematik derslerine bakmam gerekiyor". Önceki notlardaki (özellikle [07](/posts/yz50-07-neden-gradient-descent/)) CLT referansını ispatsız bırakmıştım, burası o eksiği kapatıyor.

---

## Merkezi Limit Teoremi — Karakteristik Fonksiyon İspatı

**Bu bölüm neden var:** [07](/posts/yz50-07-neden-gradient-descent/)'de mini-batch'in varyansı için "anket analojisi" kullanılmıştı ve CLT ispatsız geçilmişti. Bu bölüm o boşluğu kapatıyor — çünkü "neden karekök" sorusunun cevabı bu ispatın içinde.

### Kurulum

| Sembol | Anlamı |
|---|---|
| $X_1,\dots,X_m$ | Bağımsız, aynı dağılımdan (i.i.d.) rastgele değişkenler |
| $\mu, \sigma^2$ | X'in ortalaması ve varyansı |
| $Y = (X-\mu)/\sigma$ | Standartlaştırılmış X — ortalaması 0, varyansı 1 |
| $Z_m = \frac{1}{\sqrt{m}}\sum_i Y_i$ | Standartlaştırılmış toplam — CLT'nin iddiası bunun N(0,1)'e yakınsadığı |

### Karakteristik Fonksiyon

$$\varphi_Y(t) = \mathbb{E}[e^{itY}]$$

**Semboller:** $\varphi_Y$ = $Y$'nin karakteristik fonksiyonu — bir dağılımın Fourier dönüşümü, dağılımı **tek anlamlı** (bire bir) belirliyor · i = hayali birim · t = gerçel bir parametre (fonksiyonun kendi değişkeni, X ile karıştırma).

Bağımsızlık sayesinde toplamın karakteristik fonksiyonu, ayrı ayrı fonksiyonların çarpımı:

$$\varphi_{Z_m}(t) = \left[\varphi_Y\left(\frac{t}{\sqrt{m}}\right)\right]^m$$

### Taylor Açılımı

$\varphi_Y(s)$'yi $s=0$ civarında aç ($\mathbb{E}[Y]=0$, $\mathbb{E}[Y^2]=\mathrm{Var}[Y]=1$ olduğundan):

$$\varphi_Y(s) = 1 + is\cdot\mathbb{E}[Y] - \frac{s^2}{2}\mathbb{E}[Y^2] + o(s^2) = 1 - \frac{s^2}{2} + o(s^2)$$

$s = t/\sqrt{m}$ yerine koy:

$$\varphi_{Z_m}(t) = \left[1 - \frac{t^2}{2m} + o\left(\frac{1}{m}\right)\right]^m$$

### Limit

Klasik analiz limiti (e^x'in tanımıyla aynı aile): $\lim_{m\to\infty}(1-a/m)^m = e^{-a}$. Buradan:

$$\lim_{m\to\infty} \varphi_{Z_m}(t) = e^{-t^2/2}$$

Bu, **standart normal dağılımın karakteristik fonksiyonunun ta kendisi.** Karakteristik fonksiyonlar dağılımı tek anlamlı belirlediği için (Lévy'nin süreklilik teoremi), $Z_m \to N(0,1)$. **CLT ispatlandı.**

![CLT yakınsaması — çarpık bir dağılımdan normale](/yz50/clt-yakinsama.gif)

**Neden dağılımın şekli önemsiz:** ispatta $X$ hakkında sadece $\mu$ ve $\sigma^2$ (ilk iki moment) kullanıldı — daha yüksek momentler o(s^2) içinde kayboldu. X ister çarpık ister çok-modlu olsun, ortalama/varyans sonlu olduğu sürece toplamı Gauss'a yakınsıyor.

![CLT — çarpık dağılımdan örneklem ortalamaları normale yakınsıyor](/yz50/clt-yakinsama.svg)

Yukarıdaki, gerçek bir çarpık dağılımdan (üstel dağılım) rastgele üretilmiş veriyle çizildi — m=1'de dağılımın kendisi kadar çarpık, m=5'te belirginleşmeye başlıyor, m=30'da neredeyse tam simetrik/normal. Tam olarak ispatın söylediği şey.

**ML bağlantısı:** mini-batch'in ortalama gradient'i da ($\nabla C_x$'lerin ortalaması) — $\nabla C_x$'in kendi dağılımı ne olursa olsun — m yeterince büyükse yaklaşık Gauss dağılımlı davranıyor. Bu, "gradient gürültüsü" hakkındaki birçok teorik ML analizinin (optimizer tasarımı dahil) neden Gauss gürültüsü varsaydığının gerekçesi.

---

## Hangi İstatistik Derslerine/Konularına Bakmalısın

**Bu bölüm neden var:** Roadmap'in Faz 3.0'ı "istatistik hatırlatması" diyor ama hangi alt başlıkların gerçekten gerektiği belirsizdi. Buradaki liste, bu notlarda fiilen kullanılan konulardan çıkarıldı — genel bir müfredat değil.

Roadmap'in kendi Faz 3.0 Matematik Hatırlatma bölümü zaten "probability & istatistik: dağılımlar, Bayes, hipotez testi" diyor — bugünkü konuşmalar bunun **hangi alt-başlıklarının** senin için somut/gerekli olduğunu netleştirdi:

1. **Örnekleme teorisi (sampling theory):** yerine koyarak vs koymadan örnekleme, sonlu popülasyon düzeltmesi — mini-batch seçiminin doğrudan temeli.
2. **Tahmin ediciler (estimator theory):** yansızlık (unbiasedness), tahmin edicinin varyansı, **Bessel düzeltmesi** (neden n-1) — [07](/posts/yz50-07-neden-gradient-descent/)'teki σ-vs-s ayrımının kaynağı.
3. **Merkezi Limit Teoremi** — az önce ispatladığımız şey, standart bir "Probability Teorisi" veya "Mathematical Statistics" dersinin çekirdek konusu.
4. **Güven aralıkları, t-dağılımı vs normal dağılım** — bilinen $\sigma$ ile tahmini $s$ arasındaki farkın pratik sonucu, küçük örneklem durumunda neden t-dağılımı kullanıldığı.
5. **Maximum Likelihood Estimation (MLE)** — [07](/posts/yz50-07-neden-gradient-descent/) Kısım 3'te değindiğimiz, gradient descent'in çoğu zaman MLE'yi çözmenin aracı olması.

**Nereden:** StatQuest (zaten de Faz 3.0 için önerili) bu konuların çoğunu (CLT, tahmin ediciler, t-dağılımı) kısa/net videolarla anlatıyor — hatırlatma formatına uygun. Daha formal/ispatlı bir kaynak istersen, herhangi bir "Mathematical Statistics" ders notu (örn. Casella & Berger'in *Statistical Inference* kitabı, lisans/yüksek lisans standart referansı) bu ispatların tam hallerini içeriyor.

## Hangi Matematik Derslerine/Konularına Bakmalısın

1. **Çok değişkenli kalkülüs:** partial derivative, gradient, chain rule — zaten [12](/posts/yz50-12-backpropagation-calculus/)'da yoğun kullandık.
2. **Lineer cebir:** matrix çarpımı, matrix derivative'leri (Normal Denklem'in türetilmesi buna dayanıyor), matrix tersi — roadmap'in Faz 3.0'ında zaten var.
3. **Optimizasyon teorisi:** convexity/non-convexity, local vs global minimum, Hessian'ın rolü (pozitif tanımlı matrix kavramı — Hessian'ın eigenvalue'ları pozitifse minimum garantili) — Faz 3.0'daki "optimizasyon" maddesinin somutlaşmış hali.
4. **Fourier analizi / karmaşık sayılar (hafif):** CLT'nin ispatı karakteristik fonksiyon ($e^{itX}$) kullanıyor — tam bir "Fourier analizi dersi" gerekmiyor, ama Euler formülü (e^(iθ)=cosθ+isinθ) ve temel karmaşık sayı aritmetiğine aşinalık, ispatı rahat takip etmek için yeterli.

**Özet:** roadmap'in Faz 3.0'ı zaten doğru başlıkları listelemiş — bugünkü derinlemesine konuşma, o başlıkların **hangi spesifik alt-konularının** (Bessel düzeltmesi, CLT'nin ispatı, örnekleme teorisi, matrix derivative'leri) senin ML anlayışın için somut/gerekli olduğunu gösterdi. Faz 3'e geldiğinde bu notu tekrar aç, 30-60 dakikalık hatırlatma molanı bu listeye göre şekillendir.

---

## Bağlantı

**Bağlantılı:** [Hafta 2 — Neden Gradient Descent? İstatistiksel ve Matematiksel Temeller](/posts/yz50-07-neden-gradient-descent/)
