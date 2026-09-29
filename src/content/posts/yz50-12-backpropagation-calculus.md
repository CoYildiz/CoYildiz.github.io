---
title: "12 · Backpropagation Calculus"
published: 2026-09-29
description: "Zincir kuralının ağ üzerindeki tam muhasebesi — her kısmi türevin hangi terimden geldiği, indisleriyle birlikte."
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

Kaynak: [3blue1brown.com/lessons/backpropagation-calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus), orijinal video [YouTube'da](https://www.youtube.com/watch?v=tIeHLnjs5U8). **Bu, YZ50 Hafta 2'nin kendi kaynak listesindeki 4. madde** — resmi materyal, serinin sonuncusu. Önceki: [11 · Backpropagation Sezgisi](/posts/yz50-11-backpropagation-sezgisi/). **Bu not, `Value.backward()`'ının içinde tam olarak ne olduğunu birebir açıklıyor — Hafta 2'nin en kritik notu.**

---

## Basit Ağ — Layer Başına 1 Nöron

**Bu bölüm neden var:** Gerçek bir ağın indeksleri ($j$, $k$, layer numarası) formülü okunamaz hale getiriyor. Önce her layer'da tek nöron olan halde zinciri çıkarıyoruz; genel hal sadece bu formüle indeks eklemekten ibaret olacak.

Önce en basit hal: her layer'da tek nöron. 3 weight, 3 bias — hedef, her birinin cost'u ne kadar etkilediğini anlamak.

### Notasyon

| Sembol | Anlamı |
|---|---|
| $L$ | Son layer (üst simge, **üs değil** — sadece "hangi layer" indeksi) |
| $a^{(L)}$ | Son layer'daki nöronun aktivasyonu |
| $a^{(L-1)}$ | Bir önceki layer'daki nöronun aktivasyonu |
| $y$ | Bu örnek için istenen (doğru) çıktı |
| $w^{(L)}, b^{(L)}$ | Son layer'a ait weight ve bias |
| $\sigma$ | Sigmoid/ReLU gibi nonlineer aktivasyon fonksiyonu |

### Temel Tanımlar

Tek bir örnek için cost:

$$C_0 = (a^{(L)} - y)^2$$

**Semboller:** $C_0$ = **tek bir** eğitim örneğinin cost'u (alt simge 0, "bu belirli örnek" anlamında, tüm veri setinin ortalaması olan $C$'den farklı).

Son nöronun aktivasyonu, ağırlıklı toplam + bias'ın aktivasyon fonksiyonundan geçirilmiş hali:

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)} \qquad a^{(L)} = \sigma(z^{(L)})$$

**Semboller:** $z^{(L)}$ = aktivasyondan **önceki** ham ağırlıklı toplam (kendi adı var çünkü ayrı ayrı derivative almamız gerekecek).

**Bağımlılık zinciri:** $w^{(L)}, a^{(L-1)}, b^{(L)} \to z^{(L)} \to a^{(L)} \to (\text{y ile birlikte}) \to C_0$. Bu zincir, `Value` sınıfının computation graph'ının ta kendisi — her ok bir `_backward()` çağrısı.

![Bağımlılık zinciri / computation graph](/yz50/computation-graph.svg)

## İlk Derivative'i Hesaplamak — Zincir Kuralı

**Bu bölüm neden var:** `w`'yi oynattığında cost'un ne kadar değiştiğini doğrudan hesaplayamıyoruz, çünkü `w` cost'a doğrudan girmiyor — arada `z` ve `a` var. Chain rule tam olarak bu "arada aracılar var" durumunu çözen araç.

Hedef: $\partial C_0/\partial w^{(L)}$ — $w^{(L)}$'e küçük bir dokunuşun $C_0$'ı ne kadar etkilediği.

$$\frac{\partial C_0}{\partial w^{(L)}} = \frac{\partial z^{(L)}}{\partial w^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial C_0}{\partial a^{(L)}}$$

**Semboller:** her kesir, zincirdeki **bir adımlık** hassasiyet — "w'ye dokunursam z ne kadar değişir", "z değişirse a ne kadar değişir", "a değişirse $C_0$ ne kadar değişir." Üçünü çarpmak, uçtan uca hassasiyeti veriyor (klasik chain rule, dw/dw'nin ara adımlarda "sadeleşmesi" gibi düşün).

### Üç Alt Derivative

$$\frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)} \qquad \frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'(z^{(L)}) \qquad \frac{\partial C_0}{\partial a^{(L)}} = 2(a^{(L)}-y)$$

**Semboller:** $\sigma'(z^{(L)})$ = aktivasyon fonksiyonunun derivative'i, $z^{(L)}$ noktasında değerlendirilmiş (senin Hafta 2'de tanh için elle hesapladığın şey — $1-\tanh^2(z)$ — burada $\sigma'$ yerine geçen tam olarak o).

**Yorum (iki tanesi çok önemli):**
- **$\partial z/\partial w = a^{(L-1)}$:** weight'in etkisi, önceki nöron **ne kadar aktifse o kadar güçlü** — Hebbian paralelinin matematiksel kanıtı.
- **$\partial C_0/\partial a^{(L)} = 2(a^{(L)}-y)$:** hata ne kadar büyükse (çıktı hedeften ne kadar uzaksa), küçük bir dokunuşun bile cost'u o kadar çok değiştirdiği — büyük hatalar, büyük düzeltme sinyali üretiyor.

### Hepsini Birleştirmek

$$\frac{\partial C_0}{\partial w^{(L)}} = a^{(L-1)} \cdot \sigma'(z^{(L)}) \cdot 2(a^{(L)}-y)$$

Bu, gradient vektörünün **tek bir girdisi** — sadece bu bir örnek, sadece bu bir weight için.

## Tüm Eğitim Verisi İçin Ortalama

$$C = \frac{1}{n}\sum_{k=0}^{n-1} C_k \qquad \frac{\partial C}{\partial w^{(L)}} = \frac{1}{n}\sum_{k=0}^{n-1} \frac{\partial C_k}{\partial w^{(L)}}$$

**Semboller:** $n$ = toplam eğitim örneği sayısı · $C_k$ = k'ıncı örneğin cost'u · bu, [08](/posts/yz50-08-stochastic-gradient-descent/)'de zaten gördüğün (1/n)Σ formülünün ta kendisi, burada tek bir weight'e uygulanmış hali.

## Bias Derivative'i — Neredeyse Aynı

$$\frac{\partial C_0}{\partial b^{(L)}} = \frac{\partial z^{(L)}}{\partial b^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial C_0}{\partial a^{(L)}}$$

Tek fark: $\partial z^{(L)}/\partial b^{(L)} = 1$ (z = w·a+b'de b'nin katsayısı zaten 1). Yani:

$$\frac{\partial C_0}{\partial b^{(L)}} = 1 \cdot \sigma'(z^{(L)}) \cdot 2(a^{(L)}-y)$$

Weight formülünden sadece $a^{(L-1)}$ çarpanı eksik — bias derivative'i daha basit.

## Önceki Layer'lara Yayılma — Asıl "Backprop"

**Bu bölüm neden var:** Buraya kadar yalnızca **son** layer'ın parametrelerinin derivative'ini aldık. Ağın geri kalanına nasıl ulaşılacağı — ve yöntemin adının neden "backpropagation" olduğu — burada çıkıyor.

Önceki layer'ın weight/bias'larına ulaşmak için önce **o layer'ın aktivasyonuna göre hassasiyeti** bulmamız lazım:

$$\frac{\partial C_0}{\partial a^{(L-1)}} = \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial C_0}{\partial a^{(L)}} \qquad \text{burada } \frac{\partial z^{(L)}}{\partial a^{(L-1)}} = w^{(L)}$$

**Semboller:** $\partial z^{(L)}/\partial a^{(L-1)} = w^{(L)}$ çünkü $z^{(L)} = w^{(L)} a^{(L-1)}+b^{(L)}$ — $a^{(L-1)}$'in katsayısı $w^{(L)}$.

Bu aktivasyona **doğrudan** dokunamayız — ama bu değeri bilmek, chain rule'u **bir layer daha geriye** uygulamamızı sağlıyor: $a^{(L-1)}$ kendi $w^{(L-1)}, b^{(L-1)}$'ine bağlı, onlar için de aynı chain rule tekrar uygulanıyor. **İşte "backpropagation" kelimesinin geldiği yer** — hassasiyeti çıktıdan girdiye doğru, layer layer geriye taşıyoruz.

## Çok Nöronlu Genel Ağ

**Bu bölüm neden var:** Tek nöronlu hal gerçek bir ağ değil. Genelleme iki şey getiriyor: indeksler (mekanik) ve **çoklu yol** (asıl yeni fikir, `Value` sınıfındaki `+=`'nin sebebi).

Gerçekçi bir ağda (senin `Layer`/`MLP (Multi-Layer Perceptron)`'in gibi) her layer'da birden fazla nöron var — notasyona iki indeks daha eklenir:

| Sembol | Anlamı |
|---|---|
| $j$ | Layer $L$'deki nöron indeksi |
| $k$ | Layer $L-1$'deki nöron indeksi |
| $a_j^{(L)}$ | $L$ layer'ındaki $j$'inci nöronun aktivasyonu |
| $w_{jk}^{(L)}$ | $L-1$'deki $k$'ıncı nörondan $L$'deki $j$'inci nörona giden weight (sıra önemli — j önce, k sonra, weight matrix'i gösterimiyle aynı) |

$$z_j^{(L)} = \sum_k w_{jk}^{(L)} a_k^{(L-1)} + b_j^{(L)} \qquad C_0 = \sum_j (a_j^{(L)} - y_j)^2$$

Weight derivative'i formülü **birebir aynı**, sadece indeksli:

$$\frac{\partial C_0}{\partial w_{jk}^{(L)}} = a_k^{(L-1)} \cdot \sigma'(z_j^{(L)}) \cdot 2(a_j^{(L)}-y_j)$$

### Asıl Yeni Şey — Çoklu Yol Etkisi

Tek-nöronlu halde her önceki-layer nöronunun sadece **bir** sonraki nörona etkisi vardı. Şimdi L-1'deki bir nöron, L layer'ındaki **her** nörona bağlı — yani cost'u **birden fazla yoldan** etkiliyor. Bu yolların hepsinin etkisi **toplanmalı**:

$$\frac{\partial C_0}{\partial a_k^{(L-1)}} = \sum_j w_{jk}^{(L)} \cdot \sigma'(z_j^{(L)}) \cdot 2(a_j^{(L)}-y_j)$$

**Semboller:** $\sum_j$ — bu toplamın yeni olan kısmı: $L$ layer'ındaki **her** $j$ nöronunun bu $k$ nöronuna olan katkısını topluyor. Bu, [11](/posts/yz50-11-backpropagation-sezgisi/)'teki "10 çıktı nöronunun rekabet eden istekleri toplanıyor" cümlesinin birebir matematiği.

---

## Neden **geriye**? İleriye doğru da yapılabilirdi

**Bu bölüm neden var:** Chain rule bir çarpım zinciri ve çarpma birleşmeli (associative) — yani
bu çarpımları soldan sağa da, sağdan sola da yapabilirsin. İkisi de **aynı sayıyı** verir. O
halde neden herkes geriye doğru yapıyor? Cevap matematikte değil **maliyette**, ve backprop'un
varlık sebebi bu.

> **Hatırlatman gereken şey:** matrix çarpımının birleşme özelliği, ve çarpım sırasının işlem
> sayısını değiştirdiği (matrix zinciri çarpımı problemi). $(AB)C$ ile $A(BC)$ aynı sonucu verir
> ama maliyetleri farklı olabilir.

Zinciri tek bir weight için yaz:

$$\frac{\partial C_0}{\partial w} = \frac{\partial z}{\partial w}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial C_0}{\partial a}$$

**İleri mod (soldan başla):** $\partial z/\partial w$'den başlayıp çıktıya doğru ilerlersin. Bu
şekilde bir geçişte **tek bir girdinin** (tek bir $w$'nin) bütün çıktılara etkisini bulursun.

**Geri mod (sağdan başla):** $\partial C_0/\partial a$'dan başlayıp girdilere doğru inersin. Bir
geçişte **tek bir çıktının** bütün girdilere olan hassasiyetini bulursun.

Maliyet tablosu — $n$ parametre, $m$ çıktı:

| | Bir geçişte ne bulur | Toplam maliyet |
|---|---|---|
| **İleri mod** | bir **girdinin** etkisi | $n$ geçiş gerekir |
| **Geri mod** | bir **çıktının** hassasiyeti | $m$ geçiş gerekir |

Sinir ağı eğitiminde $n$ = parametre sayısı (13.002, ya da milyarlar), $m$ = **1** çünkü cost tek
bir skaler. Yani:

$$\frac{\text{ileri mod maliyeti}}{\text{geri mod maliyeti}} = \frac{n}{m} = \frac{13002}{1}$$

**Geri mod tek geçişte bütün derivative'leri veriyor.** İleri mod aynı sonucu 13.002 geçişte verirdi —
ki bu, [01 · Lineer Regresyon](/posts/yz50-01-lineer-regresyon/)'daki koordinat aramasının ve numerical derivative'in
tam olarak düştüğü tuzak. Backprop o tuzaktan kaçmanın adı.

**Bedeli var:** geri mod, ileri geçişteki **ara değerleri saklamak** zorunda ($z$, $a$, her layer'da).
İleri mod bunlara ihtiyaç duymaz. Yani backprop hesabı bellekle satın alıyor — derin ağlarda
belleğin neden dolduğunun ve "gradient checkpointing" diye bir tekniğin neden var olduğunun sebebi
tam olarak bu.

**Tersi de doğru:** çıktı sayısı girdi sayısından çoksa ileri mod kazanır. Bu yüzden otomatik
derivative kütüphanelerinde iki mod da var; sinir ağı eğitimi sadece $m=1$ olduğu için geri modun
uç örneği.

---

## Ek Not — Gradient mı, Jacobian mı?

**Bu bölüm neden var:** Literatürde ikisi de geçiyor ve karıştırılıyor. Ayrımı bilmek PyTorch'un `autograd` dokümantasyonunu okurken gerekiyor — orada her şey "Jacobian-vector product" diye anlatılıyor.

$\nabla C$ (13.002 boyutlu) teknik olarak bir **gradient**, Jacobian değil — çünkü $C: \mathbb{R}^{13002} \to \mathbb{R}$, yani **skaler-değerli** bir fonksiyon (13.002 sayı yiyor, 1 sayı veriyor). Jacobian, **vektör-değerli** fonksiyonlar için tanımlı (girdi de çok, çıktı da çok).

Ama Jacobian'lar backprop'un **içinde**, layer layer gerçekten var: $z^{(L)} = W^{(L)}a^{(L-1)}+b^{(L)}$ haritasının $a^{(L-1)}$'e göre Jacobian'ı tam olarak **$W^{(L)}$'nin kendisi** ($\partial z_j/\partial a_k = w_{jk}$). σ'nin Jacobian'ı ise **köşegen** bir matrix ($\text{diag}(\sigma'(z))$) — her çıktı sadece kendi girdisine bağlı olduğu için. Yukarıdaki $\sum_j$ formülü, matrix dilinde şu şekilde yazılabilir:

$$\frac{\partial C_0}{\partial a^{(L-1)}} = (W^{(L)})^T \cdot \text{diag}(\sigma'(z^{(L)})) \cdot \frac{\partial C_0}{\partial a^{(L)}}$$

Yani backprop = her layer'da o layer'ın **yerel Jacobian'ını**, gelen gradient **vektörüyle** çarpıp bir öncekine aktarmak (bir "Jacobian-vector product" zinciri). PyTorch'un `autograd`'ı da tam olarak bunu yapıyor. Ama zincirin **en ucunda** (tüm layer'lardan geçtikten sonra) elde edilen şey, $C$'nin kendisi skaler olduğu için, bir **vektör** ($\nabla C$) — matrix değil.

---

## Sonuç ve `Value.backward()` ile Bağlantı

Bu üç formül (weight, bias, önceki-layer-aktivasyonu) — senin `Value` sınıfının her bir işlem türü (toplama, çarpma, tanh) için yazdığın `_backward()` fonksiyonlarının **matematiksel kaynağı.** Micrograd'daki her `Value` node'u tam olarak burada gösterilen şeyi yapıyor: kendi yerel derivative'ini (∂z/∂w gibi) hesaplayıp, üstten gelen gradientla (∂C/∂a gibi) çarpıp, girdilerine geri gönderiyor — çoklu-yol durumunda (bir `Value` birden fazla yerde kullanıldıysa) gradient'leri **topluyor**, tam yukarıdaki $\sum_j$ gibi. **Bu senin Görev 3'ün ("ters topolojik sırayla gradient toplama") tam matematiksel gerekçesi.**

## Cost Fonksiyonu Var mı Yok mu — Zincirin Uzunluğu Değişiyor (2026-08-31)

Yukarıdaki formülde ($\partial C_0/\partial w^{(L)} = a^{(L-1)} \cdot \sigma'(z^{(L)}) \cdot 2(a^{(L)}-y)$) üç çarpan var, ama Hafta 2'nin `noron()` fonksiyonunda `o.backward()` çağırdığında (gerçek bir cost fonksiyonu hiç yoktu) **üçüncü çarpan hiç mevcut değildi.**

**Sebep:** `o.backward()` çağırınca root, `o`'nun (aktivasyon çıktısının) kendisi — `self.grad(o) = 1`, ve hemen sonrasında doğrudan $\sigma'(z)$'ye geçiliyor. Ama gerçek eğitimde `C.backward()` çağırırsın — root artık `C`, ve `o`'ya inmeden önce **$\partial C_0/\partial a^{(L)}$ halkası araya giriyor** (yukarıdaki tabloda zaten var olan üçüncü çarpan). `noron()`'da bu halka basitçe **yoktu**, zincir bir adım kısaydı.

**Somut sayılarla (n=3.0, tanh, $y=1$ varsayımsal bir MSE cost eklenirse):**

| | Cost yok (`noron()`'da yaptığın) | Cost var (MSE, y=1) |
|---|---|---|
| Zincirin başı | self.grad(o) = 1 | self.grad(C) = 1, sonra $\partial C/\partial o = 2(o-y)$ = -0.009890 |
| $\tanh'(n)$ (aktivasyonun kendi derivative'i) | 0.009866 | 0.009866 — **birebir aynı** |
| n'e ulaşan nihai gradient | 0.009866 | -0.0000976 |

$\tanh'(n)$ (aktivasyonun o noktadaki kendi davranışı) **iki durumda da birebir aynı sayı** — değişen, cost fonksiyonunun var olup olmadığı ve varsa hangisi olduğu. Cost yoksa zincir $\sigma'(z)$'de başlıyor; cost varsa önce $\partial C_0/\partial a^{(L)}$ (yukarıdaki tabloda "2(a-y)" olarak zaten yazılı) çarpanı ekleniyor, sonuç tamamen değişiyor (işaret bile değişebiliyor).

**Bağlantı:** Görev 5'e geçince (`Neuron`/`Layer`/`MLP` + gerçek training loop), ilk kez `backward()`'ı bir cost fonksiyonunun üzerinde çağıracaksın — o zaman bu üçüncü çarpan gerçekten devreye girecek, ve MSE/BCE (Binary Cross-Entropy) gibi hangi cost'u seçtiğin (bkz. [15](/posts/yz50-15-aktivasyon-loss-eslesmesi/)) sonucu doğrudan etkileyecek.

**Ek kaynaklar (kaynağın önerdiği, ihtiyaç olursa):** [Nielsen'in kitabı, Bölüm 2](http://neuralnetworksanddeeplearning.com/chap2.html) (kodu  altında), [colah'ın backprop yazısı](http://colah.github.io/posts/2015-08-Backprop/), ileri seviye için Goodfellow/Bengio/Courville'in *Deep Learning* kitabı (Faz 4 seviyesi).

**Bağlantılı:** [11 · Backpropagation Sezgisi](/posts/yz50-11-backpropagation-sezgisi/)
