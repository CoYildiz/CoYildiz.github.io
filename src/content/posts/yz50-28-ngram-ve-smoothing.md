---
title: "28 · N-gram Dil Modelleri ve Smoothing"
published: 2026-10-01
description: "Markov varsayımı, perplexity'nin 'ortalama kaç seçenek' yorumu ve sıfır olasılığı çözen smoothing yöntemleri."
tags:
  - YZ50
  - Dil Modeli
  - N-gram
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

**Bu not neden var:** [16 · Loss, NLL ve Cross-Entropy](/posts/yz50-16-loss-nll-cross-entropy/) perplexity'yi
loss'un okunabilir hali olarak tanıtıyordu, ama üç şey eksik kalmıştı: **Markov varsayımının**
ne olduğu, perplexity'nin **"dallanma faktörü"** yorumu, ve eğitim verisinde hiç görülmemiş
dizilimlerin yarattığı **sıfır olasılık** problemi ile çözümleri. Bu not o üç boşluğu dolduruyor.

Kaynak: Jurafsky & Martin, *Speech and Language Processing* (3. baskı taslağı),
[Bölüm 3 — N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf).
Aşağıdaki formüller ve sayılar o bölümden alındı; perplexity örneği yeniden hesaplanarak
doğrulandı.

---

## 1. Markov varsayımı

Bir dil modelinin işi şu: geçmiş verildiğinde bir sonraki kelimenin olasılığını söylemek.
Tam hâliyle bu, cümlenin başından o ana kadarki **bütün** kelimelere koşullu bir olasılık:

$$
P(w_n \mid w_1, w_2, \ldots, w_{n-1})
$$

Bunu doğrudan saymak imkânsız — "bugün hava çok güzel olduğu için dışarı" dizisinin eğitim
verisinde kaç kez geçtiğini sayamazsın, çünkü büyük ihtimalle hiç geçmemiştir.

**Markov varsayımı** bu çıkmazı şöyle kesiyor: geçmişin tamamı yerine **son birkaç kelimeye**
bak, gerisini at.

![Markov varsayımı: bağlam penceresi](/yz50/markov-baglam-penceresi.png)

| Model | Neye bakıyor | Formül |
|---|---|---|
| **Unigram** | hiçbir şeye | $P(w_n)$ |
| **Bigram** | son 1 kelimeye | $P(w_n \mid w_{n-1})$ |
| **Trigram** | son 2 kelimeye | $P(w_n \mid w_{n-2}, w_{n-1})$ |
| **n-gram** | son $n-1$ kelimeye | $P(w_n \mid w_{n-N+1:n-1})$ |

Kitabın tanımı: *"Markov models are the class of probabilistic models that assume we can predict
the probability of some future unit without looking too far into the past."*

Olasılıklar basitçe sayarak bulunuyor — **maximum likelihood estimation**:

$$
P(w_n \mid w_{n-1}) = \frac{C(w_{n-1} w_n)}{C(w_{n-1})}
$$

Yani "bu ikili kaç kez geçti / ilk kelime kaç kez geçti". [21 · Bigram'ın Sinir Ağı
Hali](/posts/yz50-21-onehot-ve-nn-bigram/) notunda bunun sayım tablosu hâlini zaten kurmuştuk.

### Güçlü ve zayıf yanları

| | |
|---|---|
| **Güçlü** | Eğitim = saymak. Çok hızlı, yorumlanabilir, az veriyle bile bir şey veriyor |
| **Güçlü** | Her tahminin nereden geldiği görülebiliyor — hangi sayımdan çıktığı belli |
| **Zayıf** | Pencere dışındaki bağlamı **tamamen** kaybediyor. "Bugün aldığım kitabı okudum" cümlesinde "kitabı" ile "okudum" arası uzaksa bigram bunu göremiyor |
| **Zayıf** | $n$ büyüdükçe veri ihtiyacı üstel artıyor; çoğu n-gram hiç görülmüyor |
| **Zayıf** | Görülmemiş dizilim olasılığı **tam sıfır** yapıyor — bölüm 3 |

> **Sayısal not:** Olasılıklar çarpıldığı için uzun cümlelerde çarpım alt taşmaya (underflow)
> gidiyor. Pratikte hep **log olasılıkların toplamı** olarak hesaplanıyor — aynı sebeple
> [16](/posts/yz50-16-loss-nll-cross-entropy/) notunda log'a geçmiştik.

## 2. Perplexity — "ortalama kaç seçenek arasında kararsızım"

[16](/posts/yz50-16-loss-nll-cross-entropy/) notunda perplexity'yi cross-entropy'nin üssü olarak görmüştük.
SLP3 ona çok daha sezgisel bir isim veriyor: **ağırlıklı ortalama dallanma faktörü**
(weighted average branching factor).

Kitabın örneği bunu tek bakışta anlatıyor. Üç renkli yapay bir dil düşün: `red`, `green`, `blue`.
Test dizisi: **red red red red blue**.

![Perplexity = ağırlıklı dallanma faktörü](/yz50/perplexity-dallanma.png)

**Model A** — üç renk de eşit olası ($1/3$):

$$
\text{perplexity}_A = \left(\tfrac{1}{3}\right)^{-1} = 3
$$

Üç seçenek var, hiçbiri diğerinden olası değil, model her adımda **3 seçenek arasında**
kararsız. Perplexity tam da bu: 3.

**Model B** — `red` çok olası ($0.8$), diğerleri $0.1$:

$$
\text{perplexity}_B = 0.04096^{-1/5} = 1.89
$$

**Seçenek sayısı hâlâ 3.** Ama model çoğu zaman "red" diyebildiği için pratikte sanki
**1.89 seçenek** arasında kararsız. "Ağırlıklı" dallanma faktörü bu demek.

Bu iki sayıyı yeniden hesapladım: **A = 3.0000**, **B = 1.8946** — kitabın verdiği 3 ve 1.89
ile uyuşuyor (`img/make_ngram_figures.py`).

**Düşük perplexity daha iyi model.** Ve dikkat: perplexity ancak **aynı sözlük** üzerinde
karşılaştırılabilir; farklı sözlükle eğitilmiş iki modelin perplexity'si kıyaslanamaz.

## 3. Sıfır olasılık problemi

Eğitim verisinde hiç geçmemiş bir ikili, MLE'ye göre olasılığı **tam sıfır** alıyor. Bunun iki
sonucu var ve ikincisi yıkıcı:

1. O dizilim asla üretilemiyor.
2. Test cümlesinin olasılığı bir çarpım olduğu için, **tek bir sıfır bütün cümleyi sıfırlıyor** —
   ve perplexity tanımsız hale geliyor (sıfırın tersi).

Bu, "modelin biraz hatalı olması" değil, **ölçemez hale gelmesi** demek.

Çözüm ailesinin adı **smoothing** (ya da discounting). Kitabın tanımı net:
*"Smoothing algorithms shave off a bit of probability mass from some more frequent events and
give it to unseen events."* Yani sıfırdan bir şey yaratılmıyor — **sık görülenden alınıp
görülmeyene veriliyor**.

### Laplace (add-one) smoothing

En basiti: normalize etmeden önce bütün sayımlara 1 ekle.

$$
P_{\text{Laplace}}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}w_n) + 1}{C(w_{n-1}) + V}
$$

$V$ = sözlük boyutu. Paydaya $V$ ekleniyor çünkü her kelimeye 1 eklendi, toplam $V$ arttı.

![Smoothing, sık görülenden kütle alıp görülmeyene veriyor](/yz50/smoothing-kutle-aktarimi.png)

Küçük bir örnekle (5 kelimelik sözlük, sayımlar 40/8/2/0/0):

| Kelime | Sayım | MLE | Laplace | add-k ($k{=}0.1$) |
|---|---|---|---|---|
| the | 40 | **0.800** | 0.746 | 0.794 |
| a | 8 | 0.160 | 0.164 | 0.160 |
| cat | 2 | 0.040 | 0.055 | 0.042 |
| dog | 0 | **0.000** | **0.018** | **0.002** |
| zebra | 0 | **0.000** | **0.018** | **0.002** |

Sıfırlar kalktı — ama bedeli görünüyor: `the`'nin olasılığı 0.800'den 0.746'ya düştü.
Kitabın değerlendirmesi açık: *"Laplace smoothing does not perform well enough to be used in
modern n-gram models"* — ama baseline olarak ve metin sınıflandırmada hâlâ kullanışlı.

### Add-k

Laplace'ın sorunu "1" eklemenin çok fazla olması. Add-k daha az ekliyor:

$$
P_{\text{add-}k}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}w_n) + k}{C(w_{n-1}) + kV}
$$

Tablodaki son sütun bunu gösteriyor: $k = 0.1$ ile `the` neredeyse hiç cezalanmıyor (0.794)
ama `dog` yine de sıfırdan kurtuluyor (0.002). $k$'nın kendisi bir hiperparametre —
**devset üzerinde** seçiliyor ([26](/posts/yz50-26-hiperparametre-ve-early-stopping/)).

### Interpolation — mertebeleri harmanlamak

Farklı fikir: trigram güvenilmezse, bigram ve unigram'ın bildiğini kullan. Üçünü **karıştır**:

$$
\hat{P}(w_n \mid w_{n-2} w_{n-1}) = \lambda_1 P(w_n) + \lambda_2 P(w_n \mid w_{n-1}) + \lambda_3 P(w_n \mid w_{n-2} w_{n-1})
$$

$\lambda$'lar **1'e toplanmak zorunda** — yani bu bir ağırlıklı ortalama.

$\lambda$'lar nereden geliyor? Sayımlardan değil: **held-out** bir külliyattan. Kitabın ifadesi:
*"We do so by choosing the λ values that maximize the likelihood of the held-out corpus."*
Yani n-gram olasılıkları sabitlenir, sonra held-out setin olasılığını en büyük yapan $\lambda$'lar
aranır (bir yolu EM algoritması — Jelinek & Mercer, 1980).

Daha gelişmiş hâlinde $\lambda$'lar **bağlama koşullu**: belirli bir bigram için sayımlar
güvenilirse, ona dayanan trigram'a daha yüksek ağırlık verilir.

### Stupid backoff

Harmanlamak yerine **sırayla geri çekilmek**: aradığın n-gram'ın sayımı sıfırsa bir alt
mertebeye in, orada da sıfırsa daha alta — sayımı olan bir bağlam bulana kadar.

$$
S(w_i \mid w_{i-N+1:i-1}) =
\begin{cases}
\dfrac{\text{count}(w_{i-N+1:i})}{\text{count}(w_{i-N+1:i-1})} & \text{sayım} > 0 \\[2ex]
\lambda\, S(w_i \mid w_{i-N+2:i-1}) & \text{aksi halde}
\end{cases}
$$

Geri çekilme unigram'da bitiyor: $S(w) = \text{count}(w)/N$.
Brants ve ark. (2007) **$\lambda = 0.4$** değerinin iyi çalıştığını bulmuş.

**Adındaki "stupid" boşuna değil:** bu yöntem gerçek bir olasılık dağılımı üretmiyor. Yüksek
mertebeli olasılıklardan kütle kesilmediği (discount yapılmadığı) için toplam 1 etmiyor —
kitap bu yüzden $P$ değil **$S$** harfiyle yazıyor. Çok büyük külliyatlarda ucuzluğu bu kusuru
telafi ettiği için kullanılıyor.

### Dördünün karşılaştırması

| Yöntem | Güçlü | Zayıf |
|---|---|---|
| **Laplace** | En basit, tek satır; sıfırları anında kaldırıyor | Sık görülenleri aşırı cezalandırıyor; modern n-gram modellerinde yetersiz |
| **Add-k** | Cezanın şiddetini ayarlayabiliyorsun | $k$ için devset gerekiyor; hâlâ iyi çalışmıyor |
| **Interpolation** | Bütün mertebelerin bilgisini kullanıyor; gerçek olasılık dağılımı üretiyor | $\lambda$'ları öğrenmek için ayrı held-out külliyat gerekiyor |
| **Stupid backoff** | Çok ucuz, çok büyük veride pratik | **Olasılık dağılımı değil**; perplexity gibi olasılık temelli ölçütlerle kullanılamaz |

## 4. Bunun YZ50'deki karşılığı

Bu notun konusu doğrudan Hafta 3'ün bigram çalışmasına bağlanıyor:

- Sayım tablosu ve ondan olasılık üretme → [21 · Bigram'ın Sinir Ağı Hali](/posts/yz50-21-onehot-ve-nn-bigram/)
- Olasılık dağılımından örnekleme → [19 · Sampling Döngüsü](/posts/yz50-19-sampling-dongusu/),
  [20 · torch.multinomial](/posts/yz50-20-multinomial/)
- Perplexity'nin cross-entropy ile ilişkisi → [16 · Loss, NLL ve Cross-Entropy](/posts/yz50-16-loss-nll-cross-entropy/)

Ve **smoothing'in sinir ağındaki karşılığı**: Karpathy'nin bigram çalışmasında sayım tablosuna
eklenen `+1` (model smoothing) ile sinir ağı tarafındaki **regularization** aynı işi yapıyor —
ikisi de dağılımı tek bir noktada yoğunlaşmaktan alıkoyup düzleştiriyor. Ağırlık cezalarının
nasıl çalıştığı [25 · L1 ve L2 Regularization](/posts/yz50-25-regularization-l1-l2/) notunda.

## 5. Özet

- **Markov varsayımı:** tüm geçmiş yerine son $n-1$ kelime. n-gram modelini mümkün kılan şey bu.
- **MLE:** olasılık = sayım / sayım. Eğitim, saymaktan ibaret.
- **Perplexity:** ağırlıklı ortalama dallanma faktörü — "ortalama kaç seçenek arasında
  kararsızım". Düşük iyi. Aynı sözlük olmadan karşılaştırılamaz.
- **Sıfır olasılık** sadece hata değil, ölçümü **imkânsız** kılıyor (çarpım sıfırlanıyor).
- **Smoothing** sıfırdan olasılık yaratmıyor; sık görülenden alıp görülmeyene veriyor.
- Laplace basit ama kaba · add-k ayarlanabilir · interpolation gerçek dağılım üretiyor ·
  stupid backoff ucuz ama dağılım değil.

## Kaynaklar

- Jurafsky & Martin — [*Speech and Language Processing*, Bölüm 3: N-gram Language
  Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (bu notun ana kaynağı; formüller ve
  renk dili örneği oradan)
- Brants ve ark. (2007) — stupid backoff ve $\lambda = 0.4$
- Jelinek & Mercer (1980) — interpolation $\lambda$'larının EM ile öğrenilmesi
- Bu nottaki grafikler ve yeniden hesaplanan perplexity: `img/make_ngram_figures.py`

---

**Bağlantılı:** [16 · Loss, NLL ve Cross-Entropy](/posts/yz50-16-loss-nll-cross-entropy/) · [21 · Bigram'ın Sinir Ağı Hali](/posts/yz50-21-onehot-ve-nn-bigram/) · [19 · Sampling Döngüsü](/posts/yz50-19-sampling-dongusu/) · [20 · torch.multinomial](/posts/yz50-20-multinomial/) · [25 · L1 ve L2 Regularization](/posts/yz50-25-regularization-l1-l2/)
