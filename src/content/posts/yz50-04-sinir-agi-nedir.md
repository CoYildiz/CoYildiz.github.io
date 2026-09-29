---
title: "04 · Sinir Ağı Nedir?"
published: 2026-09-29
description: "3Blue1Brown Video 1'in notları: bir sinir ağı gerçekte hangi fonksiyonu kuruyor, weight ve bias ne işe yarıyor, katmanlar neyi temsil ediyor."
tags:
  - YZ50
  - Sinir Ağları
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

Kaynak: [3blue1brown.com/lessons/neural-networks](https://www.3blue1brown.com/lessons/neural-networks) (yazılı adaptasyon), orijinal video [YouTube'da](https://www.youtube.com/watch?v=aircAruvnKk).

Örnek problem boyunca: 28×28 piksellik el yazısı rakam görüntülerini (0-9) tanıyan bir ağ.

---

## Nöron = Sayı Tutan Bir Şey

**Bu bölüm neden var:** "Nöron" kelimesi biyolojiden geldiği için kafa karıştırıyor. Buradaki nöron bir hücre değil, sadece bir sayı tutan kutu — önce bunu netleştirmeden gerisi anlaşılmıyor.

**Nöron**, 0.0 ile 1.0 arasında bir sayı (**activation**) tutan bir şey. Bir sinir ağı, birbirine bağlı bir sürü nörondan ibaret. Nöron "yanık" gibi düşün — activation yüksekse parlak, düşükse sönük.

## Ağın Yapısı (rakam tanıma örneği)

- **Input layer:** 784 nöron (28×28 = 784 piksel). Her nöronun activation'ı, o pikselin parlaklığı (0.0 = siyah, 1.0 = beyaz).
- **Output layer:** 10 nöron, her biri bir rakama (0-9) karşılık geliyor. Activation = "ağ bu görüntünün bu rakam olduğuna ne kadar inanıyor".
- **Hidden layers:** Girdi ile çıktı arasında, bu örnekte 2 layer × 16 nöron. Sayı seçimi keyfi (16, ekrana sığsın diye seçilmiş) — asıl önemli olan bu ara layer'ların *ne yaptığı*.

## Neden Layer'lar Kullanılır

**Bu bölüm neden var:** Ağı neden katmanlara bölüyoruz, tek büyük bir fonksiyon neden olmuyor? Bu bölümdeki cevap bir **umut** — ve [05](/posts/yz50-05-ag-analizi-ve-sinirlari/)'te o umudun gerçekleşmediği görülecek. İkisini birlikte okumak gerekiyor.

Umut şu: her layer, bir öncekinden daha soyut bir şeyi tanısın.

- 2. layer → **kenarlar (edges)** tanısın
- 3. layer → kenarlardan oluşan **desenler/döngüler (loops)** tanısın
- 4. layer (output) → desenlerin hangi rakama karşılık geldiğine karar versin

Yani: piksel → kenar → desen → rakam. Bu, sadece rakam tanımaya özgü değil — konuşma tanımada da benzer bir hiyerarşi var: ham ses → fonem → hece → kelime → cümle. **Layerlı yapı, zor bir problemi küçük, yönetilebilir alt-problemlere bölüyor.**

(Not: Ağın gerçekten böyle mi çalıştığı ayrı bir soru — bu sadece bizim *umudumuz*, eğitim (training) videosunda tekrar ele alınacak.)

## Layer'lar Arası Bilgi Akışı — Ağırlıklı Toplam (Weighted Sum)

**Bu bölüm neden var:** Ağın yaptığı işin tamamı bu tek işlem — girdileri weight'lerle çarpıp toplamak. Geri kalan her şey (bias, aktivasyon, katman) bunun üstüne ekleniyor.

Bir sonraki layer'daki her nöron, bir önceki layer'daki **tüm** nöronlara bağlı — ama her bağlantının bir **weight'i (weight)** var. Weight, "bu bağlantının ne kadar güçlü/önemli olduğunu" temsil ediyor: pozitif weight "önceki nöron açıksa bu da açık olsun" der, negatif weight tam tersini der.

Bir nöronun ham girdisi (henüz [0,1]'e sıkıştırılmamış), önceki layer'daki tüm activation'ların ağırlıklı toplamı:

$$z = w_1 a_1 + w_2 a_2 + \dots + w_n a_n$$

Bu toplam **herhangi bir reel sayı** olabilir — ama biz activation'ın [0, 1] arasında olmasını istiyoruz. Bunun için bir "sıkıştırma" fonksiyonuna ihtiyaç var (aşağıda: sigmoid).

**Örnek — kenar tespiti mantığı:** Bir nöronun, görüntünün belirli bir bölgesindeki bir kenarı tespit etmesini istiyorsan: o bölgedeki piksellere karşılık gelen weight'leri pozitif yap (parlaksa aktive etsin), etrafındaki (kenarın dışındaki) piksellere negatif weight ver (parlaksa bastırsın), geri kalan tüm weight'leri ~0 yap. Sonuç: sadece o spesifik ince kenar aydınlıkken, çevresi karanlıkken nöron güçlü aktive olur.

## Bias

**Bu bölüm neden var:** Ağırlıklı toplam tek başına nöronu "ne zaman ateşleyeceğini" belirleyemiyor — eşik bilgisi eksik. Bias tam olarak o eşiği veriyor. [01](/posts/yz50-01-lineer-regresyon/) ve [02](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/)'de aynı `b` parametresi olarak zaten karşına çıkmıştı.

Bazen bir nöronun sadece ağırlıklı toplam 0'dan büyük olduğunda değil, **belirli bir eşiği** (mesela 10) geçtiğinde anlamlı şekilde aktive olmasını isteriz. Bunun için ağırlıklı toplama bir sayı daha ekleriz — **bias**:

$$z = (w_1 a_1 + w_2 a_2 + \dots + w_n a_n) + b$$

Weight'ler "bu nöron hangi pikseli/deseni arıyor" sorusunu cevaplıyor, bias ise "bu toplam ne kadar büyük olmalı ki nöron gerçekten önemsesin" sorusunu cevaplıyor.

## Sigmoid Squishification

**Bu bölüm neden var:** Ağırlıklı toplam sınırsız bir sayı üretiyor ama "aktivasyon" 0-1 arası olmalı. Sigmoid bu geçişi yapıyor — ve neden başka bir fonksiyon değil sorusu [13](/posts/yz50-13-aktivasyon-fonksiyonlari/)'ün konusu.

Ham toplamı `z`'yi [0, 1] aralığına sıkıştırmak için kullanılan klasik fonksiyon: **sigmoid**, $\sigma$ ile gösterilir.

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Davranışı: çok negatif `z` → 0'a yakın, çok pozitif `z` → 1'e yakın, `z=0` civarında düzgün bir geçiş (S-eğrisi).

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
```

*(Bu, evrensel/standart bir matematik fonksiyonu — herhangi bir ders kitabında aynı şekilde geçer. Görev 1'in istediği "tek nöron forward pass" ise bundan fazlası: ağırlıklı toplam + bias + bu sigmoid'i bir araya getiren fonksiyon — onu senin tasarlaman gereken kısım.)*

Bir nöronun tam formülü, önceki layer'daki tüm `n` nöron için:

$$a = \sigma\left(\sum_{i=1}^{n} w_i a_i + b\right)$$

![Tek nöron diyagramı](/yz50/tek-noron.svg)

## Kompakt Gösterim — Matrix Çarpımı

**Bu bölüm neden var:** Tek tek nöron yazmak 13.002 parametreli bir ağda imkânsız. Matrix gösterimi sadece kısaltma değil, kodun gerçekten böyle yazılmasının sebebi — [02](/posts/yz50-02-hiperuzay-cok-degiskenli-regresyon/)'nin şekil disiplini buraya bağlanıyor.

Bütün bir layer'ın nöronlarını tek tek hesaplamak yerine, matrix/vektör gösterimiyle tek seferde ifade edilir:

- Önceki layer'ın activation'ları → bir **sütun vektörü** $\vec{a}^{(0)}$
- Tüm weight'ler → bir **matrix** $W$ (her satır, bir sonraki layer'daki bir nörona giden tüm weight'ler)
- Biaslar → bir **vektör** $\vec{b}$

$$\vec{a}^{(1)} = \sigma(W \vec{a}^{(0)} + \vec{b})$$

($\sigma$ burada vektörün her elemanına ayrı ayrı uygulanıyor.) Bu gösterim hem yazması kısa hem de pratikte çok daha hızlı — kütüphaneler matrix çarpımını agresif şekilde optimize ediyor (bu yüzden Görev 1-2'de "kütüphanesiz" yazman isteniyor, önce çıplak mantığı elle kurman için).

## Ağ Sadece Bir Fonksiyon

**Bu bölüm neden var:** Bütün karmaşıklığın altında tek bir matematiksel nesne var: 784 sayı alıp 10 sayı veren bir fonksiyon. Bu çerçeve, "öğrenme" kelimesini gizemli olmaktan çıkarıyor — öğrenmek, o fonksiyonun parametrelerini ayarlamak demek.

Sonuçta bütün ağ: 784 sayı içeri giriyor, 10 sayı dışarı çıkıyor. Karmaşık ama nihayetinde tek bir matematiksel fonksiyon — bu örnekte **13.002 parametre** (weight + bias) ile parametrelenmiş:

![Ağ mimarisi — 784 → 16 → 16 → 10](/yz50/ag-mimarisi.svg)

- Girdi → gizli layer 1: `784×16` weight + `16` bias
- Gizli layer 1 → gizli layer 2: `16×16` weight + `16` bias
- Gizli layer 2 → çıktı: `16×10` weight + `10` bias
- Toplam: `12544 + 256 + 160 = 12960` weight, `16+16+10 = 42` bias → **13002**

**"Öğrenme" (bir sonraki video), bu 13.002 sayı için doğru değerleri bulma sürecidir** — bunları elle ayarlamak yerine.

---

**Bağlantılı:** [06 · Gradient Descent](/posts/yz50-06-gradient-descent/) (henüz yok · izlenince eklenecek)
