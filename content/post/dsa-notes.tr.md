---
title: DSA Notlarım
date: 2026-07-22
lastmod: 2026-07-22
description: Joining Data
tags:
  - Notes
  - DSA
categories:
  - Notes
draft: false
---

## Veri Yapıları ve Algoritmalar (DSA) Notları

> Bu notları kendim yazıp Claude'a düzenlettim, o yüzden hatalar olabilir.
## İçindekiler

1. [Big-O Notasyonu](#1-big-o-notasyonu)
2. [Diziler (Arrays)](#2-diziler-arrays)
3. [Bağlı Listeler (Linked List)](#3-ba%C4%9Fl%C4%B1-listeler-linked-list)
4. [Yığın (Stack)](#4-y%C4%B1%C4%9F%C4%B1n-stack)
5. [Kuyruk (Queue)](#5-kuyruk-queue)
6. [Hash Table](#6-hash-table)
7. [Ağaçlar (Trees)](#7-a%C4%9Fa%C3%A7lar-trees)
    - [7.1 Trie (Prefix Tree)](#71-trie-prefix-tree--%C3%B6nek-a%C4%9Fac%C4%B1)
8. [Heap (Öncelik Kuyruğu)](#8-heap-%C3%B6ncelik-kuyru%C4%9Fu)
9. [Graf (Graph)](#9-graf-graph)
10. [Recursion (Özyineleme)](#10-recursion-%C3%B6zyineleme)
11. [Arama Algoritmaları](#11-arama-algoritmalar%C4%B1)
12. [Sıralama Algoritmaları](#12-s%C4%B1ralama-algoritmalar%C4%B1)
13. [Divide and Conquer](#13-divide-and-conquer-böl-ve-fethet)
14. [Dynamic Programming (DP)](#14-dynamic-programming-dp)
15. [Greedy Algoritmalar](#15-greedy-algoritmalar)
16. [Backtracking](#16-backtracking)
17. [Bit Manipülasyonu](#17-bit-manip%C3%BClasyonu)
18. [Heap Sort](#18-heap-sort)
19. [AVL Tree](#19-avl-tree)
20. [B-Tree Ailesi (B-Tree, B+/B- Tree, 2-3 Tree)](#20-b-tree-ailesi-b-tree-bb--tree-2-3-tree)
21. [Yönlü ve Yönsüz Graf](#21-y%C3%B6nl%C3%BC-ve-y%C3%B6ns%C3%BCz-graf-directed--undirected-graph)
22. [En Kısa Yol Algoritmaları (Dijkstra, Bellman-Ford, A*)](#22-en-k%C4%B1sa-yol-algoritmalar%C4%B1-dijkstra-bellman-ford-a)
23. [Minimum Spanning Tree (Prim's, Kruskal's)](#23-minimum-spanning-tree-prims-ve-kruskals-algoritmas%C4%B1)
24. [Segment Tree](#24-segment-tree)
25. [Fenwick Tree (BIT)](#25-fenwick-tree-binary-indexed-tree---bit)
26. [Disjoint Set / Union-Find](#26-disjoint-set--union-find)
27. [Suffix Tree ve Suffix Array](#27-suffix-tree-ve-suffix-array)
28. [Skip List](#28-skip-list)
29. [İndeksleme Teknikleri (ISAM vb.)](#29-indeksleme-teknikleri-isam-linear-indexing-tree-based-indexing)
30. [Problem Çözme Teknikleri / Pattern'lar](#30-problem-%C3%A7%C3%B6zme-teknikleri--patternlar)
31. [Özet Karşılaştırma Tabloları](#31-%C3%B6zet-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1rma-tablolar%C4%B1)

---

## 1. Big-O Notasyonu

Big-O, bir algoritmanın girdi boyutu (`n`) büyüdükçe **zaman** veya **alan (bellek)** ihtiyacının nasıl büyüdüğünü ifade eden matematiksel bir üst sınırdır. Amaç donanımdan bağımsız, "büyüme hızına" odaklanan bir karşılaştırma yapmaktır.

### Yaygın karmaşıklıklar (küçükten büyüğe)

|Notasyon|Adı|Örnek|
|---|---|---|
|O(1)|Sabit|dizide index ile erişim|
|O(log n)|Logaritmik|binary search|
|O(n)|Doğrusal|listede tek tek arama|
|O(n log n)|Log-doğrusal|merge sort, quicksort (ortalama)|
|O(n²)|Karesel|iç içe iki döngü, bubble sort|
|O(2ⁿ)|Üstel|naif fibonacci (recursion)|
|O(n!)|Faktöriyel|tüm permütasyonları üretme|

```python
# O(1) - sabit zaman
def ilk_eleman(dizi):
    return dizi[0]

# O(n) - doğrusal zaman
def toplam(dizi):
    s = 0
    for x in dizi:
        s += x
    return s

# O(n^2) - karesel zaman (iç içe döngü)
def tekrar_eden_var_mi(dizi):
    n = len(dizi)
    for i in range(n):
        for j in range(i + 1, n):
            if dizi[i] == dizi[j]:
                return True
    return False

# O(log n) - her adımda problemi yarıya indirir
def binary_search(dizi, hedef):
    sol, sag = 0, len(dizi) - 1
    while sol <= sag:
        orta = (sol + sag) // 2
        if dizi[orta] == hedef:
            return orta
        elif dizi[orta] < hedef:
            sol = orta + 1
        else:
            sag = orta - 1
    return -1
```

**Ne zaman önemlidir?** `n` küçükken (ör. 100 eleman) O(n²) ile O(n log n) arasında pratikte fark hissedilmeyebilir; ama `n` milyonlara çıktığında fark saniyelerle saatler arasına dönüşür. Bu yüzden büyük veri, gerçek zamanlı sistemler ve performans kritik kod yazarken Big-O analizi şarttır.

**Güçlü yönü:** Donanımdan bağımsız, algoritmaları objektif karşılaştırma imkanı verir. **Zayıf yönü:** Sabit çarpanları ve küçük `n` değerlerindeki gerçek performansı görmezden gelir; bazen O(n²) bir algoritma küçük veri setinde O(n log n) bir algoritmadan daha hızlı çalışabilir (sabitler nedeniyle).

---

## 2. Diziler (Arrays)

Dizi, aynı türden elemanların bellekte **ardışık (contiguous)** olarak tutulduğu en temel veri yapısıdır. Python'da yerleşik `list` dinamik bir dizidir (gerektiğinde otomatik büyür).

```python
# Oluşturma ve temel işlemler
dizi = [10, 20, 30, 40, 50]

# Index ile erişim - O(1)
print(dizi[2])          # 30

# Sona ekleme - amortized O(1)
dizi.append(60)

# Belirli bir konuma ekleme - O(n) (sağdaki elemanlar kayar)
dizi.insert(1, 15)

# Değer ile silme - O(n) (arama + kaydırma)
dizi.remove(15)

# Sondan silme - O(1)
son = dizi.pop()

# Baştan silme - O(n)
ilk = dizi.pop(0)

# Arama (değerin index'i) - O(n)
idx = dizi.index(30)

# Dilimleme (slicing) - O(k), k = dilim uzunluğu
alt_dizi = dizi[1:3]
```

### Karmaşıklık Tablosu

|İşlem|Zaman|Açıklama|
|---|---|---|
|Index ile erişim|O(1)|Bellek adresi doğrudan hesaplanır|
|Sonuna ekleme|O(1)*|*Amortized; array büyümesi gerekirse O(n)|
|Başa/ortaya ekleme|O(n)|Diğer elemanlar kaydırılmalı|
|Değer arama|O(n)|Sırasız dizide doğrusal tarama|
|Silme (index bilinmiyorsa)|O(n)|Önce arama sonra kaydırma|
|Alan (space)|O(n)|n eleman için n birim yer|

### Nerede ve ne zaman tercih edilir?

- Elemanlara **sık sık index ile rastgele erişim** gerekiyorsa (ör. matris işlemleri, görüntü verisi).
- Eleman sayısı **önceden biliniyorsa** veya nadiren değişiyorsa.
- **Cache-friendly** olması gerektiğinde (ardışık bellek, CPU cache'ini iyi kullanır) — sayısal hesaplama, DSP, oyun motorları.
- Sıralı veri üzerinde binary search gibi algoritmalar çalıştırılacaksa.

### Güçlü yönleri

- O(1) rastgele erişim.
- Bellek düzeni sayesinde çok hızlı iterasyon (cache locality).
- Basit ve az bellek overhead'i (linked list'e göre).

### Zayıf yönleri

- Ortaya/başa ekleme-silme pahalıdır (O(n)).
- Sabit boyutlu dizilerde (C gibi dillerde) boyut önceden belirlenmeli; Python listesi bunu otomatik halleder ama büyüme maliyeti vardır.
- Çok büyük dizilerde ardışık bellek bloğu bulmak zor olabilir.

---

## 3. Bağlı Listeler (Linked List)

Her eleman (**node/düğüm**) kendi değerini ve bir sonraki (ve/veya bir önceki) düğüme referansı tutar. Elemanlar bellekte ardışık değildir.

```python
class Node:
    def __init__(self, veri):
        self.veri = veri
        self.sonraki = None

class BagliListe:
    def __init__(self):
        self.head = None

    def basa_ekle(self, veri):          # O(1)
        yeni = Node(veri)
        yeni.sonraki = self.head
        self.head = yeni

    def sona_ekle(self, veri):          # O(n) - tek yönlü liste, tail tutulmuyorsa
        yeni = Node(veri)
        if not self.head:
            self.head = yeni
            return
        dugum = self.head
        while dugum.sonraki:
            dugum = dugum.sonraki
        dugum.sonraki = yeni

    def sil(self, veri):                # O(n)
        dugum = self.head
        onceki = None
        while dugum:
            if dugum.veri == veri:
                if onceki:
                    onceki.sonraki = dugum.sonraki
                else:
                    self.head = dugum.sonraki
                return True
            onceki, dugum = dugum, dugum.sonraki
        return False

    def ara(self, veri):                # O(n)
        dugum = self.head
        while dugum:
            if dugum.veri == veri:
                return True
            dugum = dugum.sonraki
        return False

    def yazdir(self):
        dugum, elemanlar = self.head, []
        while dugum:
            elemanlar.append(dugum.veri)
            dugum = dugum.sonraki
        print(" -> ".join(map(str, elemanlar)))
```

Python'da `collections.deque` çift yönlü bağlı liste mantığıyla çalışır ve pratikte doğrudan linked list yazmak yerine sıkça bunun kullanılması tercih edilir.

### Karmaşıklık Tablosu

|İşlem|Zaman|Açıklama|
|---|---|---|
|Başa ekleme|O(1)|Sadece head referansı değişir|
|Sona ekleme|O(1) / O(n)|tail pointer varsa O(1), yoksa O(n)|
|Index ile erişim|O(n)|Baştan itibaren gezmek gerekir|
|Arama|O(n)|Doğrusal tarama|
|Silme (node biliniyorsa)|O(1)|Doubly linked list'te|
|Alan|O(n)|Her node + pointer overhead|

### Nerede ve ne zaman tercih edilir?

- Sık sık **başa/ortaya ekleme-silme** yapılacak ve index erişimi önemli değilse (ör. LRU cache implementasyonu, undo/redo geçmişi).
- Dizinin boyutu çok sık ve öngörülemez şekilde değişiyorsa (bellek parçalanmasını azaltır — büyük bir bloğa ihtiyaç yoktur).
- Queue/Stack gibi yapıların altyapısında.

### Güçlü yönleri

- Başta/ortada ekleme-silme dizi tabanlı yapılara göre çok daha verimli (node referansı biliniyorsa O(1)).
- Bellek, ihtiyaç oldukça dinamik olarak ayrılır; büyük ardışık blok gerekmez.

### Zayıf yönleri

- Rastgele erişim yok — index'e ulaşmak için baştan gezmek gerekir (O(n)).
- Her node için ekstra pointer belleği harcanır.
- Cache locality kötüdür (node'lar bellekte dağınık), bu yüzden pratikte dizilerden yavaş çalışabilir.

---

## 4. Yığın (Stack)

**LIFO** (Last In, First Out) prensibiyle çalışır: en son eklenen eleman ilk çıkar. Python'da genellikle `list` ile (append/pop sondan) uygulanır.

```python
yigin = []

yigin.append(1)   # push - O(1)
yigin.append(2)
yigin.append(3)

tepe = yigin[-1]   # peek - O(1)
cikan = yigin.pop()  # pop - O(1)  -> 3

print(yigin)       # [1, 2]

# Örnek kullanım: dengeli parantez kontrolü
def parantez_dengeli_mi(ifade):
    yigin = []
    eslesme = {')': '(', ']': '[', '}': '{'}
    for karakter in ifade:
        if karakter in "([{":
            yigin.append(karakter)
        elif karakter in ")]}":
            if not yigin or yigin.pop() != eslesme[karakter]:
                return False
    return not yigin
```

### Karmaşıklık Tablosu

|İşlem|Zaman|
|---|---|
|Push (ekleme)|O(1)|
|Pop (çıkarma)|O(1)|
|Peek (tepeye bakma)|O(1)|
|Arama|O(n)|
|Alan|O(n)|

### Nerede ve ne zaman tercih edilir?

- **Geri alma (undo)** işlemleri, tarayıcı geri/ileri geçmişi.
- **Fonksiyon çağrı yığını** (call stack), recursion'ın altyapısı.
- Parantez/etiket eşleştirme, ifade değerlendirme (infix→postfix dönüşümü).
- **DFS (Derinlik Öncelikli Arama)** algoritmasının iteratif implementasyonu.

### Güçlü yönleri

- Tüm temel işlemler O(1) — çok hızlı ve basit.
- Doğal olarak "geri izleme" (backtracking) mantığına uyar.

### Zayıf yönleri

- Sadece en üstteki elemana erişim var; ortadaki bir elemana ulaşmak O(n) ve yapıyı bozar.
- Rastgele erişim veya arama için uygun değildir.

---

## 5. Kuyruk (Queue)

**FIFO** (First In, First Out) prensibiyle çalışır: ilk giren ilk çıkar. Python listesiyle baştan pop yapmak O(n) olduğundan, bunun yerine **`collections.deque`** kullanılmalıdır (iki ucundan da O(1)).

```python
from collections import deque

kuyruk = deque()

kuyruk.append(1)      # enqueue - O(1)
kuyruk.append(2)
kuyruk.append(3)

ilk = kuyruk.popleft()  # dequeue - O(1)  -> 1
print(kuyruk)            # deque([2, 3])

# Öncelik kuyruğu (priority queue) için heapq kullanılır (bkz. bölüm 8)

# Örnek kullanım: BFS (Breadth-First Search)
def bfs(graf, baslangic):
    ziyaret_edildi = {baslangic}
    kuyruk = deque([baslangic])
    sira = []
    while kuyruk:
        dugum = kuyruk.popleft()
        sira.append(dugum)
        for komsu in graf.get(dugum, []):
            if komsu not in ziyaret_edildi:
                ziyaret_edildi.add(komsu)
                kuyruk.append(komsu)
    return sira
```

**Not:** `deque`, çift yönlü kuyruk (double-ended queue) olduğu için hem normal queue hem de stack olarak kullanılabilir; ayrıca **circular queue** ve **priority queue** de bu ailenin varyasyonlarıdır.

### Karmaşıklık Tablosu

|İşlem|Zaman (deque ile)|Zaman (list ile, baştan pop)|
|---|---|---|
|Enqueue (sona ekleme)|O(1)|O(1)|
|Dequeue (baştan çıkarma)|O(1)|O(n) ❌|
|Peek|O(1)|O(1)|
|Alan|O(n)|O(n)|

### Nerede ve ne zaman tercih edilir?

- **BFS** (en kısa yol, katman katman gezinme) algoritmalarında.
- **İş kuyrukları / görev zamanlama** (ör. yazıcı kuyruğu, mesaj kuyrukları — RabbitMQ, Kafka mantığı).
- Sırayla işlenmesi gereken herhangi bir akış (ör. müşteri destek talepleri).
- CPU **process scheduling** (round-robin).

### Güçlü yönleri

- Adil sıralama garantisi (ilk gelen ilk hizmet alır).
- `deque` ile iki uçtan da O(1) işlem.

### Zayıf yönleri

- Ortadaki bir elemana erişim/silme O(n).
- Sıradan `list` ile yanlış kullanılırsa (baştan pop) performans ciddi şekilde düşer — bu yaygın bir hatadır.

---

## 6. Hash Table

Anahtar-değer (key-value) çiftlerini bir **hash fonksiyonu** aracılığıyla dizi indekslerine eşleyerek saklar. Python'da yerleşik `dict` ve `set` bu yapıyı kullanır.

```python
# dict - hash table implementasyonu
telefon_rehberi = {}

telefon_rehberi["Ayşe"] = "0555-111-2233"   # ekleme - O(1) ortalama
telefon_rehberi["Mehmet"] = "0555-444-5566"

print(telefon_rehberi["Ayşe"])              # okuma - O(1) ortalama
print("Ayşe" in telefon_rehberi)            # arama - O(1) ortalama

del telefon_rehberi["Mehmet"]               # silme - O(1) ortalama

# Çakışma (collision) örneği - iki farklı anahtar aynı bucket'a düşebilir
# Python bunu "open addressing" ile içeride otomatik yönetir

# set - sadece anahtarları tutan hash table
gorulenler = set()
for sayi in [1, 2, 2, 3, 3, 3]:
    if sayi in gorulenler:      # O(1) ortalama kontrol
        print(f"{sayi} tekrar ediyor")
    gorulenler.add(sayi)

# Örnek kullanım: iki dizide ortak elemanları bulma - O(n)
def ortak_elemanlar(a, b):
    a_set = set(a)
    return [x for x in b if x in a_set]
```

### Karmaşıklık Tablosu

|İşlem|Ortalama|En Kötü Durum|
|---|---|---|
|Ekleme|O(1)|O(n) (çok fazla çakışma varsa)|
|Okuma/Arama|O(1)|O(n)|
|Silme|O(1)|O(n)|
|Alan|O(n)|O(n)|

_En kötü durum, tüm anahtarların aynı bucket'a çakışması (kötü hash fonksiyonu) durumunda oluşur; pratikte iyi bir hash fonksiyonuyla nadiren gerçekleşir._

### Nerede ve ne zaman tercih edilir?

- **Hızlı arama/varlık kontrolü** gerektiğinde (ör. "bu kullanıcı adı daha önce alınmış mı?").
- **Sayma / gruplama** işlemlerinde (kelime frekansı, gruplama by key).
- **Cache** implementasyonlarında (ör. memoization, LRU cache).
- Veritabanı indeksleme, veri tekilleştirme (deduplication).

### Güçlü yönleri

- Ortalama O(1) ekleme/okuma/silme — pratikte en hızlı arama yapılarından biri.
- Esnek anahtar tipleri (string, tuple, sayı — hashable her şey).

### Zayıf yönleri

- **Sırasız**dır (Python 3.7+ dict ekleme sırasını korur ama bu bir "sıralama" garantisi değildir, sıralı erişim için uygun değildir).
- Kötü hash fonksiyonu veya çok fazla çakışma en kötü durumda O(n)'e düşürebilir.
- Ekstra bellek overhead'i (dizilere göre daha fazla yer kaplar).

---

## 7. Ağaçlar (Trees)

Hiyerarşik bir veri yapısıdır: bir **kök (root)** düğümden başlayıp dallanan **node**'lardan oluşur. En yaygın türü **Binary Search Tree (BST)**'dir: sol alt ağaç < node < sağ alt ağaç kuralına uyar.

```python
class TreeNode:
    def __init__(self, deger):
        self.deger = deger
        self.sol = None
        self.sag = None

class BST:
    def __init__(self):
        self.root = None

    def ekle(self, deger):                      # ortalama O(log n), en kötü O(n)
        self.root = self._ekle(self.root, deger)

    def _ekle(self, node, deger):
        if node is None:
            return TreeNode(deger)
        if deger < node.deger:
            node.sol = self._ekle(node.sol, deger)
        else:
            node.sag = self._ekle(node.sag, deger)
        return node

    def ara(self, deger):                        # ortalama O(log n), en kötü O(n)
        return self._ara(self.root, deger)

    def _ara(self, node, deger):
        if node is None:
            return False
        if node.deger == deger:
            return True
        return self._ara(node.sol, deger) if deger < node.deger else self._ara(node.sag, deger)

    def inorder(self):                            # sıralı gezinme - O(n)
        sonuc = []
        def gez(node):
            if node:
                gez(node.sol)
                sonuc.append(node.deger)
                gez(node.sag)
        gez(self.root)
        return sonuc

# Kullanım
agac = BST()
for deger in [50, 30, 70, 20, 40, 60, 80]:
    agac.ekle(deger)
print(agac.inorder())       # [20, 30, 40, 50, 60, 70, 80] - sıralı!
print(agac.ara(60))         # True
```

**Not:** Dengesiz bir BST'ye sıralı veri eklenirse (ör. 1,2,3,4,5...) yapı bir bağlı listeye dönüşür ve işlemler O(n)'e düşer. Bu yüzden pratikte **AVL Tree** veya **Red-Black Tree** gibi kendini dengeleyen ağaçlar kullanılır. **Trie** (prefix tree) ise string/kelime aramalarında (otomatik tamamlama) kullanılan özel bir ağaç türüdür.

### Karmaşıklık Tablosu (dengeli BST)

|İşlem|Ortalama|En Kötü Durum (dengesiz)|
|---|---|---|
|Arama|O(log n)|O(n)|
|Ekleme|O(log n)|O(n)|
|Silme|O(log n)|O(n)|
|Alan|O(n)|O(n)|

### Nerede ve ne zaman tercih edilir?

- **Sıralı veri** üzerinde hızlı arama + ekleme + silme aynı anda gerekiyorsa (hash table sıralama sunmaz, dizi ekleme/silmede yavaştır).
- Dosya sistemleri, veritabanı indeksleri (**B-Tree**), otomatik tamamlama (**Trie**).
- Hiyerarşik veri modelleme: organizasyon şemaları, HTML DOM, karar ağaçları.
- **Heap** (bkz. bölüm 8), öncelik yönetimi için özel bir ağaç türüdür.

### Güçlü yönleri

- Dengeliyken hem arama hem ekleme hem silme O(log n) — dizi ve linked list'in en iyi yanlarını birleştirir.
- Doğal olarak sıralı gezinme (inorder traversal) sunar.

### Zayıf yönleri

- Dengesiz kalırsa performans O(n)'e düşer; dengeleme (AVL, Red-Black) ekstra karmaşıklık getirir.
- Hash table'a göre arama biraz daha yavaştır (O(log n) vs O(1)).
- Implementasyonu dizi/linked list'e göre daha karmaşıktır.

---

## 7.1 Trie (Prefix Tree / Önek Ağacı)

Trie, özellikle **string**'leri (kelimeleri) saklamak ve **ortak öneklere (prefix)** göre hızlı sorgulamak için tasarlanmış özel bir ağaç yapısıdır. Her düğüm bir karakteri temsil eder; kökten bir yaprağa/işaretli düğüme giden yol bir kelimeyi oluşturur.

```python
class TrieNode:
    def __init__(self):
        self.cocuklar = {}      # karakter -> TrieNode
        self.kelime_sonu = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def ekle(self, kelime):                    # O(k), k = kelime uzunluğu
        node = self.root
        for karakter in kelime:
            if karakter not in node.cocuklar:
                node.cocuklar[karakter] = TrieNode()
            node = node.cocuklar[karakter]
        node.kelime_sonu = True

    def ara(self, kelime):                     # O(k) - tam kelime var mı?
        node = self._prefix_node(kelime)
        return node is not None and node.kelime_sonu

    def onek_var_mi(self, onek):                # O(k) - bu önekle başlayan kelime var mı?
        return self._prefix_node(onek) is not None

    def _prefix_node(self, dizi):
        node = self.root
        for karakter in dizi:
            if karakter not in node.cocuklar:
                return None
            node = node.cocuklar[karakter]
        return node

# Kullanım
trie = Trie()
for kelime in ["kedi", "kelebek", "kelime", "kale"]:
    trie.ekle(kelime)

print(trie.ara("kedi"))          # True
print(trie.ara("kel"))           # False (tam kelime değil)
print(trie.onek_var_mi("kel"))   # True  ("kelebek", "kelime" ile eşleşir)
print(trie.onek_var_mi("ke"))    # True
print(trie.onek_var_mi("xyz"))   # False

# Örnek kullanım: otomatik tamamlama (autocomplete)
def onekle_baslayan_kelimeler(trie, onek):       # O(k + tüm eşleşen düğüm sayısı)
    node = trie._prefix_node(onek)
    if node is None:
        return []
    sonuc = []
    def dfs(node, yol):
        if node.kelime_sonu:
            sonuc.append(yol)
        for karakter, cocuk in node.cocuklar.items():
            dfs(cocuk, yol + karakter)
    dfs(node, onek)
    return sonuc

print(onekle_baslayan_kelimeler(trie, "kel"))    # ['kelebek', 'kelime']
```

### Karmaşıklık Tablosu

|İşlem|Zaman|Açıklama|
|---|---|---|
|Ekleme|O(k)|k = kelime/string uzunluğu, n'den (kelime sayısı) bağımsız|
|Tam kelime arama|O(k)||
|Önek (prefix) sorgusu|O(k)||
|Alan|O(ALPHABET_SIZE × N × K)|En kötü durumda; ortak önekler sayesinde pratikte çok daha az|

_Not: Hash table'da tam kelime araması da O(1) ortalama olabilir, ama Trie'nin asıl gücü **önek sorgularında** ortaya çıkar — hash table önek bazlı sorgu için uygun değildir._

### Nerede ve ne zaman tercih edilir?

- **Otomatik tamamlama (autocomplete)** ve arama motoru öneri sistemleri (Google arama kutusu mantığı).
- **Yazım denetimi (spell checker)** ve "bunu mu demek istediniz?" önerileri.
- **IP routing** tablolarında en uzun önek eşleştirme (longest prefix matching).
- Kelime oyunları (Boggle, Scrabble) için geçerli kelime kontrolü.
- Bir metinde birden fazla kelimeyi aynı anda arama (Aho-Corasick algoritmasının temeli).

### Güçlü yönleri

- Önek tabanlı sorgular hash table'dan çok daha verimlidir (hash table'da tüm kelimeleri tek tek taramak gerekir, Trie'de O(k)).
- Arama süresi, saklanan kelime **sayısından değil**, aranan string'in **uzunluğundan** bağımsız olarak ölçeklenir.
- Ortak önekleri paylaşan kelimeler bellek tasarrufu sağlar (ör. "kelime" ve "kelebek" "kel" düğümlerini paylaşır).

### Zayıf yönleri

- Ortak önek azsa (kelimeler birbirinden çok farklıysa) hash table'a göre **daha fazla bellek** kullanabilir (her düğüm için pointer/dict overhead'i).
- Implementasyonu basit bir hash table'a göre daha karmaşıktır.
- Sadece string/dizi benzeri sıralı veriler için anlamlıdır — genel amaçlı bir veri yapısı değildir.

---

## 8. Heap (Öncelik Kuyruğu)

Heap, her zaman **en küçük** (min-heap) veya **en büyük** (max-heap) elemana O(1)'de erişim sağlayan, tam ikili ağaç (complete binary tree) yapısıdır. Python'da `heapq` modülü **min-heap** olarak çalışır.

```python
import heapq

# Min-heap oluşturma
sayilar = [5, 1, 8, 3, 9, 2]
heapq.heapify(sayilar)          # O(n) - yerinde (in-place) heap'e çevirir

heapq.heappush(sayilar, 0)      # ekleme - O(log n)
en_kucuk = heapq.heappop(sayilar)  # en küçüğü çıkarma - O(log n)
print(en_kucuk)                 # 0

print(sayilar[0])               # en küçüğe bakma (peek) - O(1)

# Max-heap istenirse değerler negatiflenerek kullanılır
max_heap = [-x for x in [5, 1, 8, 3]]
heapq.heapify(max_heap)
en_buyuk = -heapq.heappop(max_heap)   # 8

# Örnek kullanım: bir listedeki en büyük k elemanı bulma - O(n log k)
def en_buyuk_k(dizi, k):
    return heapq.nlargest(k, dizi)

# Örnek kullanım: Dijkstra'nın en kısa yol algoritmasında öncelik kuyruğu
def dijkstra(graf, baslangic):
    mesafeler = {dugum: float('inf') for dugum in graf}
    mesafeler[baslangic] = 0
    pq = [(0, baslangic)]
    while pq:
        mevcut_mesafe, dugum = heapq.heappop(pq)
        if mevcut_mesafe > mesafeler[dugum]:
            continue
        for komsu, agirlik in graf[dugum].items():
            yeni_mesafe = mevcut_mesafe + agirlik
            if yeni_mesafe < mesafeler[komsu]:
                mesafeler[komsu] = yeni_mesafe
                heapq.heappush(pq, (yeni_mesafe, komsu))
    return mesafeler
```

### Karmaşıklık Tablosu

|İşlem|Zaman|
|---|---|
|Min/Max'a erişim (peek)|O(1)|
|Ekleme (push)|O(log n)|
|Çıkarma (pop)|O(log n)|
|Dizi → heap dönüşümü (heapify)|O(n)|
|Alan|O(n)|

### Nerede ve ne zaman tercih edilir?

- **Öncelik kuyruğu** gerektiren her senaryo: görev zamanlama (en yüksek öncelikli görev önce), hastane triyaj sistemleri.
- **Dijkstra** ve **Prim** gibi graf algoritmalarında en kısa yol/minimum ağırlık bulma.
- **"En büyük/küçük k eleman"** problemleri — tüm veriyi sıralamaya gerek kalmadan.
- **Heap Sort** algoritmasının temeli.

### Güçlü yönleri

- En küçük/büyük elemana anlık erişim (O(1)) ve O(log n) ile güncelleme.
- Tüm diziyi sıralamaktan (O(n log n)) çok daha verimlidir, çünkü sadece "en iyi" eleman garantisi verir, tam sıralama gerekmez.

### Zayıf yönleri

- Ortadaki rastgele bir elemanı arama O(n)'dir (heap sadece kök hakkında garanti verir).
- Tam sıralı bir çıktı istenirse (tüm elemanlar sıralı) heap tek başına yeterli değildir, ekstra pop işlemleri gerekir.

---

## 9. Graf (Graph)

Düğümler (**vertex/node**) ve bunları birbirine bağlayan kenarlardan (**edge**) oluşan yapı. Yönlü/yönsüz, ağırlıklı/ağırlıksız olabilir. Genellikle **komşuluk listesi (adjacency list)** ile temsil edilir (seyrek graflarda daha verimli).

```python
from collections import deque, defaultdict

# Komşuluk listesi ile graf temsili
graf = defaultdict(list)

def kenar_ekle(u, v, yonlu=False):
    graf[u].append(v)
    if not yonlu:
        graf[v].append(u)

kenar_ekle("A", "B")
kenar_ekle("A", "C")
kenar_ekle("B", "D")
kenar_ekle("C", "D")

# BFS - Breadth-First Search (en kısa yol - ağırlıksız graf) - O(V + E)
def bfs(graf, baslangic):
    ziyaret = {baslangic}
    kuyruk = deque([baslangic])
    sira = []
    while kuyruk:
        dugum = kuyruk.popleft()
        sira.append(dugum)
        for komsu in graf[dugum]:
            if komsu not in ziyaret:
                ziyaret.add(komsu)
                kuyruk.append(komsu)
    return sira

# DFS - Depth-First Search (recursive) - O(V + E)
def dfs(graf, dugum, ziyaret=None):
    if ziyaret is None:
        ziyaret = set()
    ziyaret.add(dugum)
    sonuc = [dugum]
    for komsu in graf[dugum]:
        if komsu not in ziyaret:
            sonuc += dfs(graf, komsu, ziyaret)
    return sonuc

print(bfs(graf, "A"))   # ['A', 'B', 'C', 'D']
print(dfs(graf, "A"))   # ['A', 'B', 'D', 'C']
```

`V` = düğüm (vertex) sayısı, `E` = kenar (edge) sayısı.

### Karmaşıklık Tablosu

|İşlem/Algoritma|Zaman|Alan|
|---|---|---|
|BFS|O(V + E)|O(V)|
|DFS|O(V + E)|O(V)|
|Dijkstra (heap ile)|O((V + E) log V)|O(V)|
|Komşuluk listesi ile kenar arama|O(derece)|O(V + E)|
|Komşuluk matrisi ile kenar arama|O(1)|O(V²)|

### Nerede ve ne zaman tercih edilir?

- **Sosyal ağlar** (arkadaşlık ilişkileri), **harita/navigasyon** (en kısa yol), **ağ topolojisi** (internet yönlendirme).
- **Bağımlılık çözümleme**: derleyiciler, paket yöneticileri (npm, pip) — **topological sort** ile.
- **Öneri sistemleri**, web sayfası sıralama (PageRank grafiği).
- BFS: en kısa yol (ağırlıksız) veya "seviye seviye" gezinme gerektiğinde. DFS: tüm yolları keşfetme, döngü tespiti, backtracking problemlerinde.

### Güçlü yönleri

- Gerçek dünyadaki ilişkisel/ağ verisini doğal şekilde modeller.
- Komşuluk listesi seyrek graflarda çok bellek-verimlidir (O(V+E)).

### Zayıf yönleri

- Yoğun (dense) graflarda komşuluk listesi/matrisi çok bellek tüketebilir (O(V²)).
- Bazı graf algoritmaları (ör. tüm çiftler en kısa yol - Floyd-Warshall O(V³)) büyük graflarda maliyetlidir.
- Implementasyon ve doğru veri yapısını (liste vs matris) seçmek, diğer yapılara göre daha fazla tasarım kararı gerektirir.

---

## 10. Recursion (Özyineleme)

Bir fonksiyonun **kendini çağırmasıdır**. Her recursive fonksiyonda mutlaka bir **temel durum (base case)** olmalıdır, aksi halde sonsuz döngüye (stack overflow) girer.

```python
# Klasik örnek: faktöriyel
def faktoriyel(n):
    if n <= 1:          # temel durum (base case)
        return 1
    return n * faktoriyel(n - 1)   # kendini çağırma

print(faktoriyel(5))    # 120

# Naif Fibonacci - O(2^n), verimsiz (tekrar tekrar aynı alt problemleri çözer)
def fib_naif(n):
    if n <= 1:
        return n
    return fib_naif(n - 1) + fib_naif(n - 2)

# Kuyruk özyinelemesi benzeri (Python TCO'yu optimize etmez, yine de örnek)
def toplam(dizi, index=0):
    if index == len(dizi):
        return 0
    return dizi[index] + toplam(dizi, index + 1)

# Recursion ile Hanoi Kuleleri - O(2^n)
def hanoi(n, kaynak, hedef, yardimci):
    if n == 1:
        print(f"{kaynak} -> {hedef}")
        return
    hanoi(n - 1, kaynak, yardimci, hedef)
    print(f"{kaynak} -> {hedef}")
    hanoi(n - 1, yardimci, hedef, kaynak)
```

**Önemli:** Python'un varsayılan recursion derinliği limiti ~1000'dir (`sys.setrecursionlimit()` ile artırılabilir ama önerilmez — çok derin recursion yerine iteratif çözüm veya `sys.setrecursionlimit` yerine explicit stack kullanmak daha güvenlidir).

### Karmaşıklık Tablosu

|Örnek|Zaman|Alan (call stack)|
|---|---|---|
|Faktöriyel|O(n)|O(n)|
|Naif Fibonacci|O(2ⁿ)|O(n)|
|Binary Search (recursive)|O(log n)|O(log n)|
|Hanoi Kuleleri|O(2ⁿ)|O(n)|

### Nerede ve ne zaman tercih edilir?

- Problem doğası gereği **kendi kendine benzer alt problemlere** ayrılabiliyorsa (ağaç/graf gezinme, böl-ve-fethet, backtracking).
- Ağaç ve graf yapılarında traversal (DFS, ağaç gezinmeleri) recursion ile çok daha okunabilir yazılır.
- Matematiksel tanımı zaten özyinelemeli olan problemler (faktöriyel, Fibonacci, Hanoi).

### Güçlü yönleri

- Kod çoğu zaman **çok daha okunabilir ve kısa** olur (özellikle ağaç/graf problemlerinde).
- Böl-ve-fethet, backtracking, dynamic programming gibi ileri tekniklerin doğal temelini oluşturur.

### Zayıf yönleri

- Her çağrı **call stack**'te yer kaplar → çok derin recursion'da `RecursionError` (stack overflow) riski.
- Naif implementasyonlarda tekrar eden alt problemler yeniden hesaplanır (bkz. Fibonacci) → memoization/DP gerekir.
- Genelde iteratif çözümlere göre biraz daha fazla overhead'e (fonksiyon çağrı maliyeti) sahiptir.

---

## 11. Arama Algoritmaları

### 11.1 Linear Search (Doğrusal Arama)

```python
def linear_search(dizi, hedef):          # O(n)
    for i, deger in enumerate(dizi):
        if deger == hedef:
            return i
    return -1
```

### 11.2 Binary Search (İkili Arama)

Sadece **sıralı** dizilerde çalışır; her adımda arama alanını yarıya indirir.

```python
def binary_search(dizi, hedef):          # O(log n)
    sol, sag = 0, len(dizi) - 1
    while sol <= sag:
        orta = (sol + sag) // 2
        if dizi[orta] == hedef:
            return orta
        elif dizi[orta] < hedef:
            sol = orta + 1
        else:
            sag = orta - 1
    return -1

# Python'un yerleşik bisect modülü ile
import bisect
dizi = [10, 20, 30, 40, 50]
idx = bisect.bisect_left(dizi, 30)       # O(log n) -> 2
```

### Karmaşıklık Tablosu

|Algoritma|En İyi|Ortalama|En Kötü|Alan|Ön Koşul|
|---|---|---|---|---|---|
|Linear Search|O(1)|O(n)|O(n)|O(1)|Yok|
|Binary Search|O(1)|O(log n)|O(log n)|O(1)|Dizi **sıralı** olmalı|

### Nerede ve ne zaman tercih edilir?

- **Linear search**: veri sırasız veya küçükse, ya da linked list gibi rastgele erişimi olmayan yapılarda.
- **Binary search**: büyük, sıralı ve durağan (sık değişmeyen) veri setlerinde — sözlük, telefon rehberi, sıralı log dosyaları, "en yakın değeri bulma" problemleri.

### Güçlü/Zayıf Yönler

- **Linear search:** Her yapıda çalışır ✅, ama büyük veride yavaş ❌.
- **Binary search:** Çok hızlı (O(log n)) ✅, ama önce sıralama gerekir (O(n log n) maliyeti) ve sadece rastgele erişimli (dizi gibi) yapılarda pratik olarak verimlidir ❌ (linked list'te binary search anlamsızdır çünkü orta elemana ulaşmak zaten O(n)).

---

## 12. Sıralama Algoritmaları

### 12.1 Bubble Sort — O(n²)

```python
def bubble_sort(dizi):
    n = len(dizi)
    for i in range(n):
        degisim_oldu = False
        for j in range(n - i - 1):
            if dizi[j] > dizi[j + 1]:
                dizi[j], dizi[j + 1] = dizi[j + 1], dizi[j]
                degisim_oldu = True
        if not degisim_oldu:      # zaten sıralıysa erken çık
            break
    return dizi
```

### 12.2 Selection Sort — O(n²)

```python
def selection_sort(dizi):
    n = len(dizi)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if dizi[j] < dizi[min_idx]:
                min_idx = j
        dizi[i], dizi[min_idx] = dizi[min_idx], dizi[i]
    return dizi
```

### 12.3 Insertion Sort — O(n²) (küçük/neredeyse sıralı veride hızlı)

```python
def insertion_sort(dizi):
    for i in range(1, len(dizi)):
        anahtar = dizi[i]
        j = i - 1
        while j >= 0 and dizi[j] > anahtar:
            dizi[j + 1] = dizi[j]
            j -= 1
        dizi[j + 1] = anahtar
    return dizi
```

### 12.4 Merge Sort — O(n log n), böl-ve-fethet, kararlı (stable)

```python
def merge_sort(dizi):
    if len(dizi) <= 1:
        return dizi
    orta = len(dizi) // 2
    sol = merge_sort(dizi[:orta])
    sag = merge_sort(dizi[orta:])
    return _birlestir(sol, sag)

def _birlestir(sol, sag):
    sonuc, i, j = [], 0, 0
    while i < len(sol) and j < len(sag):
        if sol[i] <= sag[j]:
            sonuc.append(sol[i]); i += 1
        else:
            sonuc.append(sag[j]); j += 1
    sonuc.extend(sol[i:])
    sonuc.extend(sag[j:])
    return sonuc
```

### 12.5 Quick Sort — ortalama O(n log n), yerinde (in-place)

```python
def quick_sort(dizi, dusuk=0, yuksek=None):
    if yuksek is None:
        yuksek = len(dizi) - 1
    if dusuk < yuksek:
        pivot_idx = _bolustur(dizi, dusuk, yuksek)
        quick_sort(dizi, dusuk, pivot_idx - 1)
        quick_sort(dizi, pivot_idx + 1, yuksek)
    return dizi

def _bolustur(dizi, dusuk, yuksek):
    pivot = dizi[yuksek]
    i = dusuk - 1
    for j in range(dusuk, yuksek):
        if dizi[j] <= pivot:
            i += 1
            dizi[i], dizi[j] = dizi[j], dizi[i]
    dizi[i + 1], dizi[yuksek] = dizi[yuksek], dizi[i + 1]
    return i + 1
```

### 12.6 Python'un yerleşik sıralaması (Timsort) — O(n log n), pratikte kullanılması gereken

```python
dizi = [5, 2, 8, 1, 9]
dizi.sort()                    # yerinde sıralama - O(n log n)
sirali = sorted(dizi, reverse=True)   # yeni liste döner
```

### Karşılaştırma Tablosu

|Algoritma|En İyi|Ortalama|En Kötü|Alan|Kararlı mı?|
|---|---|---|---|---|---|
|Bubble Sort|O(n)|O(n²)|O(n²)|O(1)|Evet|
|Selection Sort|O(n²)|O(n²)|O(n²)|O(1)|Hayır|
|Insertion Sort|O(n)|O(n²)|O(n²)|O(1)|Evet|
|Merge Sort|O(n log n)|O(n log n)|O(n log n)|O(n)|Evet|
|Quick Sort|O(n log n)|O(n log n)|O(n²)|O(log n)|Hayır|
|Timsort (Python `sort`)|O(n)|O(n log n)|O(n log n)|O(n)|Evet|

### Nerede ve ne zaman tercih edilir?

- **Insertion sort**: küçük veri setleri veya neredeyse sıralı veri (ör. sürekli güncellenen az sayıda öğe).
- **Merge sort**: **kararlılık (stability)** önemliyse ve en kötü durumda garantili O(n log n) isteniyorsa (ör. dış bellek sıralaması, linked list sıralama).
- **Quick sort**: ortalama durumda en hızlısı olduğundan ve az ekstra bellek kullandığından genel amaçlı sıralamada yaygındır.
- **Bubble/Selection sort**: pratikte neredeyse hiç kullanılmaz — eğitim amaçlı, algoritma mantığını öğrenmek için idealdir.
- **Gerçek hayatta**: Python'da neredeyse her zaman yerleşik `sort()`/`sorted()` (Timsort) kullanılmalıdır — elle sıralama algoritması yazmak sadece öğrenme veya çok özel bir kısıt (ör. bellek limiti, özel karşılaştırma davranışı) varsa gereklidir.

### Güçlü/Zayıf Yönler

- **Merge Sort:** Garantili O(n log n) ✅, kararlı ✅, ama O(n) ekstra bellek gerektirir ❌.
- **Quick Sort:** Yerinde (az bellek) ✅, pratikte çok hızlı ✅, ama en kötü durumda (zaten sıralı veri + kötü pivot seçimi) O(n²)'ye düşebilir ❌, kararlı değil ❌.
- **Insertion Sort:** Basit, küçük/neredeyse sıralı veride hızlı ✅, ama büyük veride O(n²) çok yavaş ❌.

---

## 13. Divide and Conquer (Böl ve Fethet)

Problemi daha küçük alt problemlere **böl**, her birini bağımsız çöz (genelde recursion ile), sonra sonuçları **birleştir**. Üç adım: **Böl (Divide) → Fethet (Conquer) → Birleştir (Combine)**.

```python
# Merge Sort zaten klasik bir divide & conquer örneğiydi (bkz. bölüm 12.4)

# Başka bir örnek: dizideki maksimum alt dizi toplamı (basitleştirilmiş D&C ile)
def maksimum_alt_dizi(dizi, sol, sag):
    if sol == sag:
        return dizi[sol]
    orta = (sol + sag) // 2
    sol_max = maksimum_alt_dizi(dizi, sol, orta)
    sag_max = maksimum_alt_dizi(dizi, orta + 1, sag)
    orta_max = _orta_gecen_max(dizi, sol, orta, sag)
    return max(sol_max, sag_max, orta_max)

def _orta_gecen_max(dizi, sol, orta, sag):
    sol_toplam, en_iyi_sol = 0, float('-inf')
    for i in range(orta, sol - 1, -1):
        sol_toplam += dizi[i]
        en_iyi_sol = max(en_iyi_sol, sol_toplam)
    sag_toplam, en_iyi_sag = 0, float('-inf')
    for i in range(orta + 1, sag + 1):
        sag_toplam += dizi[i]
        en_iyi_sag = max(en_iyi_sag, sag_toplam)
    return en_iyi_sol + en_iyi_sag

# Hızlı üs alma - O(log n), naif O(n) yerine
def hizli_us(taban, us):
    if us == 0:
        return 1
    yari = hizli_us(taban, us // 2)
    if us % 2 == 0:
        return yari * yari
    return yari * yari * taban
```

### Karmaşıklık Tablosu

|Örnek|Zaman|
|---|---|
|Merge Sort|O(n log n)|
|Binary Search|O(log n)|
|Hızlı üs alma|O(log n)|
|Maksimum alt dizi (D&C)|O(n log n)|
|Strassen matris çarpımı|O(n^2.81)|

### Nerede ve ne zaman tercih edilir?

- Problem **doğal olarak bağımsız alt problemlere ayrılabiliyorsa** ve alt problemlerin çözümü birleştirilerek ana çözüme ulaşılabiliyorsa.
- **Paralelleştirme** mümkün olduğunda (alt problemler birbirinden bağımsız olduğu için ayrı thread/process'lerde çözülebilir).
- Sıralama, arama, matris çarpımı, hızlı üs alma gibi klasik problemlerde.

### Güçlü yönleri

- Genelde naif O(n²) çözümleri O(n log n)'e indirger.
- Alt problemler bağımsız olduğundan **paralel işleme** için uygundur.

### Zayıf yönleri

- Recursion overhead'i (call stack) vardır.
- Her problem doğal olarak alt problemlere bölünemez; uygun olmayan problemde gereksiz karmaşıklık ekler.
- Birleştirme (combine) adımı bazen maliyetli olabilir (ör. merge sort'ta O(n) birleştirme).

---

## 14. Dynamic Programming (DP)

Bir problemi **örtüşen alt problemlere (overlapping subproblems)** ve **optimal alt yapıya (optimal substructure)** ayırıp, her alt problemin sonucunu **saklayarak (memoization/tabulation)** tekrar hesaplamayı önler.

### 14.1 Top-down (Memoization)

```python
def fib_memo(n, hafiza=None):        # O(n) zaman, O(n) alan
    if hafiza is None:
        hafiza = {}
    if n <= 1:
        return n
    if n in hafiza:
        return hafiza[n]
    hafiza[n] = fib_memo(n - 1, hafiza) + fib_memo(n - 2, hafiza)
    return hafiza[n]

# functools.lru_cache ile daha kısa yazım
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_lru(n):
    if n <= 1:
        return n
    return fib_lru(n - 1) + fib_lru(n - 2)
```

### 14.2 Bottom-up (Tabulation)

```python
def fib_tablo(n):                    # O(n) zaman, O(n) alan
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def fib_optimize(n):                 # O(n) zaman, O(1) alan
    if n <= 1:
        return n
    onceki, simdi = 0, 1
    for _ in range(2, n + 1):
        onceki, simdi = simdi, onceki + simdi
    return simdi

# Klasik DP problemi: 0/1 Knapsack (sırt çantası)
def knapsack(agirliklar, degerler, kapasite):     # O(n * kapasite)
    n = len(agirliklar)
    dp = [[0] * (kapasite + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(kapasite + 1):
            if agirliklar[i - 1] <= w:
                dp[i][w] = max(
                    degerler[i - 1] + dp[i - 1][w - agirliklar[i - 1]],
                    dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][kapasite]

# En uzun ortak alt dizi (Longest Common Subsequence) - O(m*n)
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

### Karmaşıklık Tablosu

|Problem|Naif (recursion)|DP ile|
|---|---|---|
|Fibonacci|O(2ⁿ)|O(n) zaman, O(n) veya O(1) alan|
|0/1 Knapsack|O(2ⁿ)|O(n · kapasite)|
|Longest Common Subsequence|O(2^(m+n))|O(m · n)|
|Longest Increasing Subsequence|O(2ⁿ)|O(n log n) (optimize edilmiş)|

### Nerede ve ne zaman tercih edilir?

- Problem **örtüşen alt problemlere** sahipse (aynı alt problem birden fazla kez çözülüyorsa) ve **optimal alt yapı** varsa (büyük problemin optimal çözümü, alt problemlerin optimal çözümlerinden oluşuyorsa).
- Optimizasyon problemleri: en kısa/uzun yol, minimum maliyet, maksimum kâr (sırt çantası, hisse senedi alım-satımı, düzenleme mesafesi - edit distance).
- Metin işleme: DNA dizisi karşılaştırma, yazım denetimi (edit distance), diff algoritmaları.

### Güçlü yönleri

- Üstel zaman karmaşıklığını (O(2ⁿ)) polinom zamana (O(n), O(n²) vb.) indirger — **muazzam** performans kazancı.
- Karmaşık optimizasyon problemlerini sistematik şekilde çözmeye imkan tanır.

### Zayıf yönleri

- **Ekstra bellek** gerektirir (memoization tablosu/dizisi) — zaman-alan trade-off'u.
- Doğru **state (durum) tanımını** bulmak zor olabilir; her problem DP'ye uygun değildir (optimal alt yapı olmayan problemlerde çalışmaz).
- Tabulation yaklaşımında doğru sırayla doldurma mantığını kurmak başlangıçta kafa karıştırıcı olabilir.

---

## 15. Greedy Algoritmalar

Her adımda, o anki **en iyi görünen (local optimum)** seçimi yaparak ilerler ve geri dönmez. Bazı problemlerde local optimum'ların toplamı global optimum'a eşittir; ama her zaman değil.

```python
# Para üstü problemi (coin change - greedy, sadece belirli para birimlerinde çalışır)
def para_ustu_greedy(tutar, paralar):        # O(n log n) sıralama + O(n)
    paralar = sorted(paralar, reverse=True)
    kullanilan = []
    for para in paralar:
        while tutar >= para:
            tutar -= para
            kullanilan.append(para)
    return kullanilan if tutar == 0 else None

print(para_ustu_greedy(63, [25, 10, 5, 1]))   # [25, 25, 10, 1, 1, 1]

# Aktivite seçme problemi (Activity Selection) - O(n log n)
def aktivite_sec(aktiviteler):     # [(baslangic, bitis), ...]
    aktiviteler = sorted(aktiviteler, key=lambda x: x[1])   # bitiş zamanına göre sırala
    secilenler = [aktiviteler[0]]
    son_bitis = aktiviteler[0][1]
    for baslangic, bitis in aktiviteler[1:]:
        if baslangic >= son_bitis:
            secilenler.append((baslangic, bitis))
            son_bitis = bitis
    return secilenler

# Huffman kodlama mantığı için heap tabanlı greedy yaklaşım da kullanılır
# Kruskal ve Prim (minimum spanning tree) da greedy algoritmalardır
```

### Karmaşıklık Tablosu

|Problem|Zaman|
|---|---|
|Para üstü (uygun sistemlerde)|O(n log n)|
|Aktivite seçme|O(n log n)|
|Kruskal (MST)|O(E log E)|
|Prim (MST, heap ile)|O(E log V)|
|Huffman kodlama|O(n log n)|

### Nerede ve ne zaman tercih edilir?

- Problem **greedy-choice property** (yerel en iyi seçim, global çözüme götürür) ve **optimal substructure** özelliklerine sahipse.
- Kaynak paylaşımı/zamanlama problemleri (aktivite seçme, görev zamanlama).
- Ağ tasarımı: **Minimum Spanning Tree** (Kruskal, Prim), sıkıştırma (Huffman kodlama).
- **Dijkstra** algoritması da (negatif ağırlık olmadığında) bir greedy yaklaşımdır.

### Güçlü yönleri

- Genelde **DP'den çok daha hızlı ve basit** — geri izleme veya tablo tutma gerekmez.
- Uygun olduğu problemlerde (matroid yapısına sahip problemler) **kanıtlanabilir optimal** sonuç verir.

### Zayıf yönleri

- **Her problemde çalışmaz** — yerel en iyi seçim her zaman global en iyiyi garanti etmez (ör. genel para üstü problemi rastgele para birimlerinde greedy ile yanlış sonuç verebilir; DP gerekir).
- Bir kez seçim yapıldıktan sonra **geri dönülmez**, bu da bazı durumlarda optimal olmayan sonuca sıkışmaya yol açar.
- Bir problemin greedy ile çözülüp çözülemeyeceğini kanıtlamak bazen DP çözmekten daha zordur.

---

## 16. Backtracking

Recursion ile **tüm olası çözümleri** deneyen, ama bir yolun çözüme götürmeyeceği anlaşılır anlaşılmaz o daldan **geri dönen (backtrack)** algoritma tekniğidir. "Deneme-yanılma + budama (pruning)" olarak düşünülebilir.

```python
# N-Queens problemi - O(N!) civarı (pruning ile pratikte çok daha az)
def n_vezir_coz(n):
    sonuclar = []
    tahta = [-1] * n     # tahta[satir] = sutun

    def guvenli_mi(satir, sutun):
        for onceki_satir in range(satir):
            onceki_sutun = tahta[onceki_satir]
            if (onceki_sutun == sutun or
                abs(onceki_sutun - sutun) == abs(onceki_satir - satir)):
                return False
        return True

    def coz(satir):
        if satir == n:
            sonuclar.append(tahta[:])
            return
        for sutun in range(n):
            if guvenli_mi(satir, sutun):
                tahta[satir] = sutun     # seçim yap
                coz(satir + 1)
                tahta[satir] = -1        # geri al (backtrack)

    coz(0)
    return sonuclar

# Permütasyonlar üretme - O(n!)
def permutasyonlar(dizi):
    sonuc = []
    def coz(gecerli, kalan):
        if not kalan:
            sonuc.append(gecerli[:])
            return
        for i in range(len(kalan)):
            gecerli.append(kalan[i])
            coz(gecerli, kalan[:i] + kalan[i + 1:])
            gecerli.pop()               # geri al
    coz([], dizi)
    return sonuc

# Sudoku çözücü, labirent problemleri, subset-sum de klasik backtracking örnekleridir
```

### Karmaşıklık Tablosu

|Problem|Zaman (en kötü durum)|
|---|---|
|N-Queens|O(N!) (pruning ile pratikte çok daha az)|
|Permütasyonlar|O(n!)|
|Subset toplamı (subset sum)|O(2ⁿ)|
|Sudoku çözücü|O(9^(boş hücre sayısı))|

### Nerede ve ne zaman tercih edilir?

- **Kısıtlama sağlama problemleri (constraint satisfaction)**: Sudoku, N-Queens, harita renklendirme.
- **Kombinatorik problemler**: tüm permütasyon/kombinasyonları üretme, subset toplamları.
- Arama uzayı büyük ama **geçersiz dallar erken tespit edilip budanabiliyorsa** (pruning ile pratik performans artar).

### Güçlü yönleri

- Kaba kuvvet (brute-force)'a göre **pruning sayesinde** çok daha verimlidir — geçersiz dallara hiç girmez.
- Karmaşık kısıtlama problemlerini sistematik ve doğru şekilde çözer; tüm geçerli çözümleri bulabilir.

### Zayıf yönleri

- En kötü durumda hâlâ **üstel zaman** karmaşıklığına sahiptir — büyük problem boyutlarında yavaş kalabilir.
- Etkili "budama (pruning)" mantığı kurmak problem-özeldir ve tasarımı zaman alabilir.
- Derin recursion nedeniyle call stack sınırlamalarına takılabilir.

---

## 17. Bit Manipülasyonu

Verileri doğrudan **bit** (0/1) seviyesinde işlemektir. Çok hızlı ve az bellek kullanan çözümler sunar.

```python
x = 5        # 0b101
y = 3        # 0b011

print(x & y)   # AND -> 1   (0b001)
print(x | y)   # OR  -> 7   (0b111)
print(x ^ y)   # XOR -> 6   (0b110)
print(~x)      # NOT -> -6
print(x << 1)  # sola kaydır (x*2)  -> 10
print(x >> 1)  # sağa kaydır (x//2) -> 2

# Bir sayının çift/tek olduğunu kontrol etme - O(1)
def cift_mi(n):
    return (n & 1) == 0

# İki sayıyı toplama işlemi olmadan XOR ile "tek sayıyı bulma" - O(n)
def tek_gecen_eleman(dizi):
    sonuc = 0
    for sayi in dizi:
        sonuc ^= sayi     # çiftler birbirini götürür
    return sonuc

print(tek_gecen_eleman([4, 1, 2, 1, 2]))   # 4

# Bir sayıdaki 1 bitlerini sayma (popcount) - O(log n)
def bit_sayisi(n):
    sayac = 0
    while n:
        sayac += n & 1
        n >>= 1
    return sayac

# Python'da doğrudan da mümkün:
print(bin(13).count('1'))   # O(log n)
```

### Karmaşıklık Tablosu

|İşlem|Zaman|
|---|---|
|AND / OR / XOR / NOT|O(1)|
|Kaydırma (shift)|O(1)|
|Bit sayma (popcount)|O(log n) veya O(1) (donanım desteğiyle)|

### Nerede ve ne zaman tercih edilir?

- **Bellek/performans kritik** sistemler: gömülü sistemler, oyun motorları, ağ protokolleri (bayrak/flag yönetimi).
- **Set/flag** yapıları: bir sayının bitlerini "açık/kapalı" olarak kullanarak birden fazla boolean durumu tek bir integer'da saklama (bitmask).
- Kriptografi, sıkıştırma algoritmaları, hash fonksiyonları.
- Klasik mülakat soruları: tekrarsız elemanı bulma, ikili sayı işlemleri, güç kontrolü (2'nin kuvveti mi?).

### Güçlü yönleri

- **Çok hızlı** (donanım seviyesinde doğrudan işlem) ve **çok az bellek** kullanır.
- Bazı problemleri (ör. tek geçen elemanı bulma) ekstra veri yapısı olmadan O(1) alanla çözer.

### Zayıf yönleri

- **Okunabilirlik düşüktür** — kod, bit manipülasyonuna aşina olmayanlar için anlaşılması zordur.
- Hata ayıklaması (debug) diğer yöntemlere göre daha zahmetlidir.
- Python gibi yüksek seviyeli dillerde performans avantajı, C/C++'a göre daha sınırlıdır (Python'da int'ler zaten obje overhead'i taşır).

---

---

## 18. Heap Sort

Heap veri yapısını (bkz. bölüm 8) kullanarak yerinde (in-place) sıralama yapar. Önce diziyi bir max-heap'e çevirir, sonra kökü (en büyük eleman) tekrar tekrar sona alıp heap'i küçülterek sıralar.

```python
def heap_sort(dizi):                          # O(n log n)
    n = len(dizi)

    def heapify(dizi, n, i):
        buyuk = i
        sol, sag = 2 * i + 1, 2 * i + 2
        if sol < n and dizi[sol] > dizi[buyuk]:
            buyuk = sol
        if sag < n and dizi[sag] > dizi[buyuk]:
            buyuk = sag
        if buyuk != i:
            dizi[i], dizi[buyuk] = dizi[buyuk], dizi[i]
            heapify(dizi, n, buyuk)

    for i in range(n // 2 - 1, -1, -1):        # max-heap oluştur - O(n)
        heapify(dizi, n, i)

    for i in range(n - 1, 0, -1):               # elemanları tek tek çıkar - O(n log n)
        dizi[0], dizi[i] = dizi[i], dizi[0]
        heapify(dizi, i, 0)

    return dizi

print(heap_sort([12, 11, 13, 5, 6, 7]))   # [5, 6, 7, 11, 12, 13]
```

### Karmaşıklık Tablosu

|Durum|Zaman|Alan|Kararlı mı?|
|---|---|---|---|
|En iyi/Ortalama/En kötü|O(n log n)|O(1)|Hayır|

### Nerede ve ne zaman tercih edilir?

- **Bellek kısıtlı** ortamlarda garantili O(n log n) sıralama gerektiğinde (merge sort'un O(n) ekstra belleğine karşı heap sort O(1) ekstra bellek kullanır).
- Gerçek zamanlı sistemlerde en kötü durum performans garantisi (quick sort'un O(n²) riskine karşı) önemliyse.

### Güçlü yönleri

- Yerinde çalışır (O(1) ekstra alan) ✅ ve en kötü durumda bile O(n log n) garantisi verir ✅.

### Zayıf yönleri

- Kararlı (stable) değildir ❌.
- Pratikte quick sort'tan genelde daha yavaştır (cache locality daha kötü) ❌.

---

## 19. AVL Tree

Kendi kendini dengeleyen bir **BST** türüdür. Her düğümde sol ve sağ alt ağaçların yükseklik farkı (**balance factor**) en fazla 1 olacak şekilde, ekleme/silme sonrası **rotasyonlarla** otomatik dengelenir. Amaç: normal BST'nin en kötü durumda O(n)'e düşme riskini ortadan kaldırmak.

```python
class AVLNode:
    def __init__(self, deger):
        self.deger = deger
        self.sol = None
        self.sag = None
        self.yukseklik = 1

def yukseklik(node):
    return node.yukseklik if node else 0

def denge_faktoru(node):
    return yukseklik(node.sol) - yukseklik(node.sag) if node else 0

def sag_rotasyon(y):
    x = y.sol
    T2 = x.sag
    x.sag = y
    y.sol = T2
    y.yukseklik = 1 + max(yukseklik(y.sol), yukseklik(y.sag))
    x.yukseklik = 1 + max(yukseklik(x.sol), yukseklik(x.sag))
    return x

def sol_rotasyon(x):
    y = x.sag
    T2 = y.sol
    y.sol = x
    x.sag = T2
    x.yukseklik = 1 + max(yukseklik(x.sol), yukseklik(x.sag))
    y.yukseklik = 1 + max(yukseklik(y.sol), yukseklik(y.sag))
    return y

def avl_ekle(node, deger):                     # O(log n) - garantili
    if node is None:
        return AVLNode(deger)
    if deger < node.deger:
        node.sol = avl_ekle(node.sol, deger)
    else:
        node.sag = avl_ekle(node.sag, deger)

    node.yukseklik = 1 + max(yukseklik(node.sol), yukseklik(node.sag))
    denge = denge_faktoru(node)

    if denge > 1 and deger < node.sol.deger:            # Sol-Sol durumu
        return sag_rotasyon(node)
    if denge < -1 and deger > node.sag.deger:            # Sağ-Sağ durumu
        return sol_rotasyon(node)
    if denge > 1 and deger > node.sol.deger:              # Sol-Sağ durumu
        node.sol = sol_rotasyon(node.sol)
        return sag_rotasyon(node)
    if denge < -1 and deger < node.sag.deger:              # Sağ-Sol durumu
        node.sag = sag_rotasyon(node.sag)
        return sol_rotasyon(node)
    return node

root = None
for deger in [10, 20, 30, 40, 50, 25]:
    root = avl_ekle(root, deger)
print(root.deger, yukseklik(root))     # AVL her zaman dengeli kalır
```

### Karmaşıklık Tablosu

|İşlem|Zaman (garantili)|
|---|---|
|Arama|O(log n)|
|Ekleme|O(log n)|
|Silme|O(log n)|
|Alan|O(n)|

### Nerede ve ne zaman tercih edilir?

- **Sık okuma / az yazma** yapılan, arama performansının kritik olduğu senaryolarda (normal BST'ye göre daha sık rotasyon yaptığı için okuma-ağırlıklı işlerde Red-Black Tree'den bile daha avantajlı olabilir).
- Veritabanı indeksleme, bellek içi sıralı veri yapıları.

### Güçlü yönleri

- **Garantili O(log n)** — normal BST'nin aksine en kötü durumda bile dengesizleşmez.
- Red-Black Tree'ye göre daha **sıkı dengelidir**, bu da aramaları biraz daha hızlandırır.

### Zayıf yönleri

- Her ekleme/silmede rotasyon kontrolü ve gerekirse rotasyon yapılması **ekstra overhead** getirir; Red-Black Tree'ye göre yazma-ağırlıklı işlerde daha yavaştır.
- Implementasyonu normal BST'ye göre belirgin şekilde daha karmaşıktır.

---

## 20. B-Tree Ailesi (B-Tree, B+/B- Tree, 2-3 Tree)

Bu üç yapı da **çok yollu (multi-way) dengeli arama ağaçları**dır — her düğüm birden fazla anahtar ve çocuk tutabilir. Amaç: disk/SSD gibi **blok tabanlı depolama**da minimum disk okuma ile arama yapmak (bir düğüm = bir disk bloğu).

### 2-3 Tree

Her düğüm ya **1 anahtar/2 çocuk** (2-node) ya da **2 anahtar/3 çocuk** (3-node) tutar. Tüm yapraklar aynı derinliktedir — her zaman mükemmel dengelidir. B-Tree'nin en basit özel halidir (order 3).

### B-Tree

Her düğüm `t-1` ile `2t-1` arası anahtar tutabilir (`t` = minimum derece). Tüm veriler her düğümde (iç düğümler dahil) saklanır.

```python
# Basitleştirilmiş B-Tree (order/derece = t) - sadece arama ve ekleme mantığı
class BTreeNode:
    def __init__(self, yaprak=True):
        self.anahtarlar = []
        self.cocuklar = []
        self.yaprak = yaprak

class BTree:
    def __init__(self, t=2):           # t = minimum derece
        self.root = BTreeNode()
        self.t = t

    def ara(self, node, anahtar):                  # O(log n)
        i = 0
        while i < len(node.anahtarlar) and anahtar > node.anahtarlar[i]:
            i += 1
        if i < len(node.anahtarlar) and node.anahtarlar[i] == anahtar:
            return True
        if node.yaprak:
            return False
        return self.ara(node.cocuklar[i], anahtar)

    def ekle(self, anahtar):                        # O(log n) (basitleştirilmiş - split mantığı gerçek B-Tree'de daha detaylıdır)
        root = self.root
        if len(root.anahtarlar) == 2 * self.t - 1:
            yeni_root = BTreeNode(yaprak=False)
            yeni_root.cocuklar.append(self.root)
            self._split_cocuk(yeni_root, 0)
            self.root = yeni_root
        self._eksik_dolu_ekle(self.root, anahtar)

    def _split_cocuk(self, ebeveyn, i):
        t = self.t
        cocuk = ebeveyn.cocuklar[i]
        yeni = BTreeNode(yaprak=cocuk.yaprak)
        ebeveyn.anahtarlar.insert(i, cocuk.anahtarlar[t - 1])
        ebeveyn.cocuklar.insert(i + 1, yeni)
        yeni.anahtarlar = cocuk.anahtarlar[t:]
        cocuk.anahtarlar = cocuk.anahtarlar[:t - 1]
        if not cocuk.yaprak:
            yeni.cocuklar = cocuk.cocuklar[t:]
            cocuk.cocuklar = cocuk.cocuklar[:t]

    def _eksik_dolu_ekle(self, node, anahtar):
        i = len(node.anahtarlar) - 1
        if node.yaprak:
            node.anahtarlar.append(None)
            while i >= 0 and anahtar < node.anahtarlar[i]:
                node.anahtarlar[i + 1] = node.anahtarlar[i]
                i -= 1
            node.anahtarlar[i + 1] = anahtar
        else:
            while i >= 0 and anahtar < node.anahtarlar[i]:
                i -= 1
            i += 1
            if len(node.cocuklar[i].anahtarlar) == 2 * self.t - 1:
                self._split_cocuk(node, i)
                if anahtar > node.anahtarlar[i]:
                    i += 1
            self._eksik_dolu_ekle(node.cocuklar[i], anahtar)

btree = BTree(t=2)
for anahtar in [10, 20, 5, 6, 12, 30, 7, 17]:
    btree.ekle(anahtar)
print(btree.ara(btree.root, 6))    # True
print(btree.ara(btree.root, 99))   # False
```

### B+ Tree

B-Tree'nin veritabanlarında en yaygın kullanılan varyantı: **tüm gerçek veriler sadece yapraklarda** tutulur, iç düğümler sadece "yönlendirme" (routing) anahtarları içerir. Ayrıca yapraklar birbirine **bağlı liste** ile bağlanır — bu da **aralık sorgularını (range query)** çok hızlandırır (ör. `WHERE yas BETWEEN 20 AND 30`).

### Karmaşıklık Tablosu

|İşlem|2-3 Tree|B-Tree / B+ Tree|
|---|---|---|
|Arama|O(log n)|O(log n) (çok daha az disk erişimi ile)|
|Ekleme|O(log n)|O(log n)|
|Silme|O(log n)|O(log n)|
|Aralık sorgusu|O(log n + k)|B+ Tree'de O(log n + k) — çok hızlı|

### Nerede ve ne zaman tercih edilir?

- **Veritabanı indeksleri** (MySQL InnoDB, PostgreSQL) ve **dosya sistemleri** (NTFS, ext4) — B+ Tree endüstri standardıdır.
- Disk/SSD gibi **yüksek gecikmeli depolama birimlerinde**, az sayıda büyük düğüm okuyarak (her düğüm = 1 disk bloğu) toplam I/O sayısını azaltmak gerektiğinde.
- 2-3 Tree: genelde eğitim amaçlı, B-Tree/AVL kavramlarını basitleştirilmiş şekilde öğretmek için kullanılır.

### Güçlü yönleri

- **Disk dostu**: yüksek dallanma faktörü (branching factor) sayesinde ağaç derinliği çok düşük kalır → az disk erişimi.
- B+ Tree'de yapraklar bağlı olduğundan **aralık sorguları** çok verimlidir.

### Zayıf yönleri

- Implementasyonu (özellikle split/merge mantığı) oldukça karmaşıktır.
- Bellek içi (RAM) kullanım için AVL/Red-Black Tree'ye göre genelde gereksiz overhead getirir — asıl avantajı disk tabanlı sistemlerde ortaya çıkar.

---

## 21. Yönlü ve Yönsüz Graf (Directed & Undirected Graph)

Bölüm 9'da graf temelleri işlenmişti; burada **yönlü (directed)** ve **yönsüz (undirected)** graf arasındaki farka odaklanıyoruz.

- **Undirected Graph:** Kenarların yönü yoktur — `A-B` kenarı hem `A→B` hem `B→A` demektir (ör. Facebook arkadaşlığı — karşılıklıdır).
- **Directed Graph (Digraph):** Kenarların yönü vardır — `A→B` sadece A'dan B'ye gidilebileceği anlamına gelir, tersi otomatik doğru değildir (ör. Twitter/X takip ilişkisi, web sayfası linkleri).

```python
from collections import defaultdict

class Graph:
    def __init__(self, yonlu=False):
        self.graf = defaultdict(list)
        self.yonlu = yonlu

    def kenar_ekle(self, u, v, agirlik=1):        # O(1)
        self.graf[u].append((v, agirlik))
        if not self.yonlu:
            self.graf[v].append((u, agirlik))

    def dongu_var_mi_yonsuz(self):                 # O(V + E) - undirected graph'ta döngü kontrolü
        ziyaret = set()
        def dfs(dugum, ebeveyn):
            ziyaret.add(dugum)
            for komsu, _ in self.graf[dugum]:
                if komsu not in ziyaret:
                    if dfs(komsu, dugum):
                        return True
                elif komsu != ebeveyn:
                    return True
            return False
        for dugum in list(self.graf):
            if dugum not in ziyaret:
                if dfs(dugum, None):
                    return True
        return False

    def dongu_var_mi_yonlu(self):                  # O(V + E) - directed graph'ta döngü kontrolü (topological sort mantığı)
        ziyaret, yigin_uzerinde = set(), set()
        def dfs(dugum):
            ziyaret.add(dugum)
            yigin_uzerinde.add(dugum)
            for komsu, _ in self.graf[dugum]:
                if komsu not in ziyaret:
                    if dfs(komsu):
                        return True
                elif komsu in yigin_uzerinde:
                    return True
            yigin_uzerinde.remove(dugum)
            return False
        for dugum in list(self.graf):
            if dugum not in ziyaret:
                if dfs(dugum):
                    return True
        return False

undirected = Graph(yonlu=False)
undirected.kenar_ekle("A", "B")
undirected.kenar_ekle("B", "C")
print(undirected.dongu_var_mi_yonsuz())    # False

directed = Graph(yonlu=True)
directed.kenar_ekle("A", "B")
directed.kenar_ekle("B", "C")
directed.kenar_ekle("C", "A")
print(directed.dongu_var_mi_yonlu())       # True (A->B->C->A döngüsü var)
```

### Karmaşıklık Tablosu

|İşlem|Undirected|Directed|
|---|---|---|
|Kenar ekleme|O(1)|O(1)|
|Döngü tespiti|O(V + E)|O(V + E)|
|BFS/DFS|O(V + E)|O(V + E)|
|Komşuluk matrisi alanı|O(V²) (simetrik)|O(V²)|

### Nerede ve ne zaman tercih edilir?

- **Undirected:** karşılıklı ilişkiler — sosyal ağ arkadaşlığı, yol/ulaşım ağları (çift yönlü yollar), elektrik devreleri.
- **Directed:** tek yönlü ilişkiler — web sayfası linkleri (PageRank), görev bağımlılıkları (derleme sırası, topological sort), takip/takipçi ilişkileri, durum makineleri (state machine).

### Güçlü/Zayıf Yönler

- **Undirected:** Modelleme daha basittir ✅, ama yön bilgisi taşınamadığı için tek yönlü ilişkileri (ör. "kim kimi takip ediyor") ifade edemez ❌.
- **Directed:** Gerçek dünyadaki asimetrik ilişkileri doğru modeller ✅, ama döngü tespiti, bağlılık (connectivity) analizi gibi işlemler undirected'e göre daha karmaşıktır (ör. "strongly connected components" kavramı sadece directed graflarda anlamlıdır) ❌.

---

## 22. En Kısa Yol Algoritmaları (Dijkstra, Bellman-Ford, A*)

### 22.1 Dijkstra'nın Algoritması

Tek bir kaynaktan tüm diğer düğümlere **en kısa yolu** bulur. Sadece **negatif olmayan** kenar ağırlıklarında doğru çalışır. Heap tabanlı implementasyonu bölüm 8'de gösterilmişti; burada tekrar özetliyoruz.

```python
import heapq

def dijkstra(graf, baslangic):                 # O((V + E) log V)
    mesafeler = {dugum: float('inf') for dugum in graf}
    mesafeler[baslangic] = 0
    pq = [(0, baslangic)]
    while pq:
        d, dugum = heapq.heappop(pq)
        if d > mesafeler[dugum]:
            continue
        for komsu, agirlik in graf[dugum]:
            yeni_d = d + agirlik
            if yeni_d < mesafeler[komsu]:
                mesafeler[komsu] = yeni_d
                heapq.heappush(pq, (yeni_d, komsu))
    return mesafeler
```

### 22.2 Bellman-Ford Algoritması

Dijkstra'dan daha yavaştır ama **negatif ağırlıklı kenarları da destekler** ve **negatif döngü tespiti** yapabilir.

```python
def bellman_ford(kenarlar, dugumler, baslangic):    # O(V * E)
    mesafeler = {dugum: float('inf') for dugum in dugumler}
    mesafeler[baslangic] = 0

    for _ in range(len(dugumler) - 1):               # V-1 kez gevşetme (relaxation)
        for u, v, agirlik in kenarlar:
            if mesafeler[u] != float('inf') and mesafeler[u] + agirlik < mesafeler[v]:
                mesafeler[v] = mesafeler[u] + agirlik

    for u, v, agirlik in kenarlar:                    # negatif döngü kontrolü
        if mesafeler[u] != float('inf') and mesafeler[u] + agirlik < mesafeler[v]:
            raise ValueError("Graf negatif ağırlıklı bir döngü içeriyor!")

    return mesafeler

kenarlar = [("A", "B", 4), ("A", "C", 1), ("C", "B", -3), ("B", "D", 2)]
print(bellman_ford(kenarlar, ["A", "B", "C", "D"], "A"))
```

### 22.3 A* Algoritması

Dijkstra'nın **heuristic (sezgisel) fonksiyonla** hızlandırılmış halidir. Her düğümü `f(n) = g(n) + h(n)` ile değerlendirir: `g(n)` = başlangıçtan o düğüme gerçek maliyet, `h(n)` = o düğümden hedefe **tahmini** maliyet (ör. Öklid/Manhattan mesafesi).

```python
import heapq

def a_yildiz(graf, baslangic, hedef, heuristic):     # O(E) civarı (iyi heuristic ile Dijkstra'dan çok daha hızlı)
    pq = [(0 + heuristic(baslangic), 0, baslangic)]
    ziyaret = set()
    g_skorlari = {baslangic: 0}

    while pq:
        _, g, dugum = heapq.heappop(pq)
        if dugum == hedef:
            return g
        if dugum in ziyaret:
            continue
        ziyaret.add(dugum)
        for komsu, agirlik in graf[dugum]:
            yeni_g = g + agirlik
            if yeni_g < g_skorlari.get(komsu, float('inf')):
                g_skorlari[komsu] = yeni_g
                f = yeni_g + heuristic(komsu)
                heapq.heappush(pq, (f, yeni_g, komsu))
    return float('inf')     # hedefe ulaşılamadı

# Örnek heuristic: ızgara (grid) üzerinde Manhattan mesafesi
def manhattan(dugum, hedef=(5, 5)):
    return abs(dugum[0] - hedef[0]) + abs(dugum[1] - hedef[1])
```

### Karşılaştırma Tablosu

|Algoritma|Zaman|Negatif Ağırlık|Heuristic|En Uygun Kullanım|
|---|---|---|---|---|
|Dijkstra|O((V+E) log V)|❌ Desteklemez|❌ Yok|Genel en kısa yol, negatif ağırlık yok|
|Bellman-Ford|O(V · E)|✅ Destekler|❌ Yok|Negatif ağırlık olabilecek graf, döngü tespiti|
|A*|Heuristic'e bağlı (iyi heuristic ile ≈ O(E))|❌ Desteklemez|✅ Var|Hedefi belli olan arama (harita/oyun)|

### Nerede ve ne zaman tercih edilir?

- **Dijkstra:** Ağ yönlendirme (routing), GPS navigasyon (negatif ağırlık olmayan yol ağları).
- **Bellman-Ford:** Döviz kuru arbitraj tespiti (negatif döngü = kâr fırsatı), ağlarda negatif maliyetli senaryolar (ör. indirimli geçişler).
- __A_:_* Oyun yapay zekâsı (pathfinding), robotik navigasyon, harita uygulamalarında **hedefi bilinen** en kısa yol aramaları — Dijkstra'dan çok daha hızlıdır çünkü hedefe doğru "yönlendirilmiş" arama yapar.

### Güçlü/Zayıf Yönler

- **Dijkstra:** Basit ve verimli ✅, ama negatif ağırlıkta yanlış sonuç verir ❌.
- **Bellman-Ford:** Negatif ağırlığı destekler ve negatif döngü tespit eder ✅, ama Dijkstra'dan belirgin şekilde daha yavaştır (O(V·E) vs O((V+E)logV)) ❌.
- __A_:_* İyi bir heuristic ile çok hızlı ✅, ama heuristic **admissible** (asla gerçek maliyeti abartmamalı) olmazsa optimal olmayan sonuç verebilir ❌, ve iyi bir heuristic tasarlamak probleme özeldir.

---

## 23. Minimum Spanning Tree: Prim's ve Kruskal's Algoritması

**Minimum Spanning Tree (MST)**: Ağırlıklı, bağlı (connected) ve yönsüz bir grafın tüm düğümlerini, **döngü oluşturmadan** ve **toplam kenar ağırlığını minimize ederek** birbirine bağlayan alt kümesidir.

### 23.1 Kruskal's Algoritması (kenar-tabanlı, Greedy)

Tüm kenarları ağırlığa göre sırala, en küçüğünden başlayarak **döngü oluşturmayan** kenarları ekle (Union-Find ile kontrol edilir — bkz. bölüm 26).

```python
def kruskal(dugumler, kenarlar):                # O(E log E)
    kenarlar = sorted(kenarlar, key=lambda x: x[2])   # (u, v, agirlik)
    ebeveyn = {dugum: dugum for dugum in dugumler}

    def bul(x):
        while ebeveyn[x] != x:
            ebeveyn[x] = ebeveyn[ebeveyn[x]]   # path compression
            x = ebeveyn[x]
        return x

    def birlestir(x, y):
        kok_x, kok_y = bul(x), bul(y)
        if kok_x == kok_y:
            return False
        ebeveyn[kok_x] = kok_y
        return True

    mst, toplam_agirlik = [], 0
    for u, v, agirlik in kenarlar:
        if birlestir(u, v):                     # döngü oluşturmuyorsa ekle
            mst.append((u, v, agirlik))
            toplam_agirlik += agirlik
    return mst, toplam_agirlik

kenarlar = [("A","B",4), ("A","C",1), ("B","C",2), ("B","D",5), ("C","D",8)]
print(kruskal(["A","B","C","D"], kenarlar))
```

### 23.2 Prim's Algoritması (düğüm-tabanlı, Greedy)

Bir başlangıç düğümünden başlar, her adımda **ağaca en yakın (en ucuz)** kenarı ekleyerek büyür (heap ile).

```python
import heapq
from collections import defaultdict

def prim(graf, baslangic):                      # O(E log V)
    ziyaret = {baslangic}
    pq = [(agirlik, baslangic, komsu) for komsu, agirlik in graf[baslangic]]
    heapq.heapify(pq)
    mst, toplam_agirlik = [], 0

    while pq and len(ziyaret) < len(graf):
        agirlik, u, v = heapq.heappop(pq)
        if v in ziyaret:
            continue
        ziyaret.add(v)
        mst.append((u, v, agirlik))
        toplam_agirlik += agirlik
        for komsu, w in graf[v]:
            if komsu not in ziyaret:
                heapq.heappush(pq, (w, v, komsu))

    return mst, toplam_agirlik

graf = defaultdict(list)
for u, v, w in [("A","B",4), ("A","C",1), ("B","C",2), ("B","D",5), ("C","D",8)]:
    graf[u].append((v, w)); graf[v].append((u, w))
print(prim(graf, "A"))
```

### Karşılaştırma Tablosu

|Algoritma|Zaman|Yaklaşım|En Uygun Graf Tipi|
|---|---|---|---|
|Kruskal|O(E log E)|Kenar bazlı, Union-Find|Seyrek (sparse) graflar|
|Prim|O(E log V) (heap ile)|Düğüm bazlı, heap|Yoğun (dense) graflar|

### Nerede ve ne zaman tercih edilir?

- **Ağ tasarımı**: elektrik şebekesi, telekomünikasyon kablo döşeme, su boru hattı — minimum maliyetle tüm noktaları bağlama.
- **Kümeleme (clustering)** algoritmalarının temelinde (MST tabanlı kümeleme).
- **Kruskal:** Kenar listesi zaten elde varsa ve graf seyrekse tercih edilir.
- **Prim:** Graf yoğunsa veya komşuluk listesi/matrisiyle çalışılıyorsa, başlangıç düğümünden büyüyerek ilerlemek daha doğal olduğunda tercih edilir.

### Güçlü/Zayıf Yönler

- **Kruskal:** Anlaşılması kolay, seyrek graflarda çok verimli ✅, ama kenarları sıralamak gerektiği için yoğun graflarda Prim'e göre yavaş kalabilir ❌.
- **Prim:** Yoğun graflarda daha verimli ✅, ama Union-Find yerine heap yönetimi gerektirdiği için implementasyonu biraz daha karmaşıktır ❌.
- Her ikisi de **greedy** algoritmalardır ve **her zaman optimal MST'yi garanti eder** (matroid teorisi sayesinde kanıtlanmıştır) ✅.

---

## 24. Segment Tree

Bir dizi üzerinde **aralık sorgularını** (range sum, range min/max vb.) ve **tekil güncellemeleri** aynı anda **O(log n)**'de yapmaya izin veren ikili ağaç yapısıdır. Naif yaklaşımda aralık sorgusu O(n), segment tree ile O(log n)'e iner.

```python
class SegmentTree:
    def __init__(self, dizi):
        self.n = len(dizi)
        self.agac = [0] * (4 * self.n)
        self._insa(dizi, 0, 0, self.n - 1)

    def _insa(self, dizi, node, sol, sag):          # O(n) - bir kere
        if sol == sag:
            self.agac[node] = dizi[sol]
            return
        orta = (sol + sag) // 2
        self._insa(dizi, 2*node+1, sol, orta)
        self._insa(dizi, 2*node+2, orta+1, sag)
        self.agac[node] = self.agac[2*node+1] + self.agac[2*node+2]

    def guncelle(self, index, deger):                # O(log n)
        self._guncelle(0, 0, self.n - 1, index, deger)

    def _guncelle(self, node, sol, sag, index, deger):
        if sol == sag:
            self.agac[node] = deger
            return
        orta = (sol + sag) // 2
        if index <= orta:
            self._guncelle(2*node+1, sol, orta, index, deger)
        else:
            self._guncelle(2*node+2, orta+1, sag, index, deger)
        self.agac[node] = self.agac[2*node+1] + self.agac[2*node+2]

    def aralik_toplami(self, l, r):                   # O(log n)
        return self._sorgu(0, 0, self.n - 1, l, r)

    def _sorgu(self, node, sol, sag, l, r):
        if r < sol or sag < l:
            return 0
        if l <= sol and sag <= r:
            return self.agac[node]
        orta = (sol + sag) // 2
        return (self._sorgu(2*node+1, sol, orta, l, r) +
                self._sorgu(2*node+2, orta+1, sag, l, r))

dizi = [1, 3, 5, 7, 9, 11]
st = SegmentTree(dizi)
print(st.aralik_toplami(1, 3))     # 3+5+7 = 15
st.guncelle(1, 10)                  # dizi[1] = 10 olarak güncelle
print(st.aralik_toplami(1, 3))     # 10+5+7 = 22
```

### Karmaşıklık Tablosu

|İşlem|Naif Dizi|Segment Tree|
|---|---|---|
|Aralık sorgusu|O(n)|O(log n)|
|Tekil güncelleme|O(1)|O(log n)|
|İnşa (build)|—|O(n)|
|Alan|O(n)|O(4n) ≈ O(n)|

### Nerede ve ne zaman tercih edilir?

- **Sık güncellenen** dizilerde **tekrarlanan aralık sorguları** gerektiğinde (ör. rekabetçi programlama, oyun skor tabloları, finansal zaman serisi analizleri — "belirli tarih aralığındaki toplam/max/min").
- Range Minimum Query (RMQ), range sum, range GCD gibi problemlerde.

### Güçlü yönleri

- Hem sorgu hem güncelleme O(log n) — **dinamik veri + sık aralık sorgusu** kombinasyonunda idealdir.
- Sum dışında min, max, GCD gibi farklı birleştirme (merge) fonksiyonlarına kolayca uyarlanabilir.

### Zayıf yönleri

- Implementasyonu prefix sum gibi basit tekniklere göre belirgin şekilde daha karmaşıktır.
- Sadece **statik boyutlu** dizilerde pratiktir; sık eleman ekleme/çıkarma (boyut değişimi) gerekiyorsa yeniden inşa gerekebilir.
- Veri **hiç güncellenmiyorsa** (sadece sorgu varsa), daha basit olan **prefix sum** dizisi (O(1) sorgu, O(n) inşa) tercih edilmelidir.

---

## 25. Fenwick Tree (Binary Indexed Tree - BIT)

Segment Tree'ye benzer bir amaca hizmet eder (aralık sorgusu + güncelleme) ama **çok daha az bellek** ve **daha basit kod** ile, sadece **prefix sum (önek toplamı)** türü işlemler için optimize edilmiştir. Bit manipülasyonundan (`i & -i` ile en sağdaki 1 bitini bulma) faydalanır.

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.agac = [0] * (n + 1)     # 1-indexed

    def guncelle(self, i, delta):                 # O(log n) - i pozisyonuna delta ekle
        i += 1                                     # 1-indexed'e çevir
        while i <= self.n:
            self.agac[i] += delta
            i += i & (-i)                           # bir sonraki sorumlu düğüme git

    def onek_toplami(self, i):                     # O(log n) - [0, i] aralığının toplamı
        i += 1
        toplam = 0
        while i > 0:
            toplam += self.agac[i]
            i -= i & (-i)
        return toplam

    def aralik_toplami(self, l, r):                 # O(log n)
        return self.onek_toplami(r) - (self.onek_toplami(l - 1) if l > 0 else 0)

dizi = [1, 3, 5, 7, 9, 11]
ft = FenwickTree(len(dizi))
for i, deger in enumerate(dizi):
    ft.guncelle(i, deger)

print(ft.aralik_toplami(1, 3))     # 3+5+7 = 15
ft.guncelle(1, 10 - 3)              # dizi[1]'i 3'ten 10'a güncelle (fark ekleniyor)
print(ft.aralik_toplami(1, 3))     # 10+5+7 = 22
```

### Karmaşıklık Tablosu

|İşlem|Fenwick Tree|Segment Tree|
|---|---|---|
|Önek toplamı sorgusu|O(log n)|O(log n)|
|Tekil güncelleme|O(log n)|O(log n)|
|İnşa|O(n log n) (tekil ekleme ile)|O(n)|
|Alan|O(n) — daha az sabit çarpan|O(4n)|
|Kod karmaşıklığı|Düşük|Orta-Yüksek|

### Nerede ve ne zaman tercih edilir?

- Sadece **toplam (sum), XOR** gibi **tersinir (invertible)** birleştirme işlemleri yeterliyse (min/max gibi tersinir olmayan işlemler için Segment Tree gerekir).
- Rekabetçi programlama, ters çevirme (inversion) sayma problemleri, dinamik sıralama/istatistik sorguları.
- Segment Tree'ye göre **daha az kod ve daha az bellek** ile aynı işi (prefix sum bağlamında) yapmak istendiğinde.

### Güçlü yönleri

- Segment Tree'ye göre **çok daha kısa ve basit** kod, daha az bellek overhead'i.
- Bit manipülasyonu sayesinde çok hızlı sabit çarpanlara sahiptir.

### Zayıf yönleri

- Sadece **prefix sum tabanlı** (toplanabilir/tersinir) işlemlerle sınırlıdır — min/max gibi sorgular için uygun değildir (Segment Tree gerekir).
- Segment Tree'ye göre **daha az esnektir** ve anlaşılması (bit trick'ler nedeniyle) ilk bakışta daha az sezgiseldir.

---

## 26. Disjoint Set / Union-Find

Elemanları **ayrık kümelere (disjoint sets)** ayırıp, iki eleman aynı kümede mi (`find`) ve iki kümeyi birleştirme (`union`) işlemlerini neredeyse O(1)'e yakın sürede yapan veri yapısıdır. **Path compression** ve **union by rank** optimizasyonlarıyla çok verimli hale gelir.

```python
class UnionFind:
    def __init__(self, elemanlar):
        self.ebeveyn = {e: e for e in elemanlar}
        self.rank = {e: 0 for e in elemanlar}

    def bul(self, x):                              # O(α(n)) ≈ O(1) - path compression ile
        if self.ebeveyn[x] != x:
            self.ebeveyn[x] = self.bul(self.ebeveyn[x])   # yol sıkıştırma
        return self.ebeveyn[x]

    def birlestir(self, x, y):                      # O(α(n)) ≈ O(1) - union by rank ile
        kok_x, kok_y = self.bul(x), self.bul(y)
        if kok_x == kok_y:
            return False                             # zaten aynı kümede -> döngü demek
        if self.rank[kok_x] < self.rank[kok_y]:
            kok_x, kok_y = kok_y, kok_x
        self.ebeveyn[kok_y] = kok_x
        if self.rank[kok_x] == self.rank[kok_y]:
            self.rank[kok_x] += 1
        return True

    def ayni_kumede_mi(self, x, y):                  # O(α(n)) ≈ O(1)
        return self.bul(x) == self.bul(y)

# Örnek kullanım: bir grafta döngü tespiti (Kruskal'da da bu şekilde kullanılıyordu)
uf = UnionFind(["A", "B", "C", "D"])
print(uf.birlestir("A", "B"))    # True  - birleşti
print(uf.birlestir("B", "C"))    # True  - birleşti
print(uf.birlestir("A", "C"))    # False - zaten aynı kümedeler -> döngü!
print(uf.ayni_kumede_mi("A", "D"))  # False
```

`α(n)`, **ters Ackermann fonksiyonu**dur — pratikte her `n` değeri için 4'ten küçüktür, yani işlemler **neredeyse O(1)** kabul edilir.

### Karmaşıklık Tablosu

|İşlem|Zaman (optimize edilmiş)|
|---|---|
|Find|O(α(n)) ≈ O(1)|
|Union|O(α(n)) ≈ O(1)|
|Aynı kümede mi kontrolü|O(α(n)) ≈ O(1)|
|Alan|O(n)|

### Nerede ve ne zaman tercih edilir?

- **Kruskal's algoritmasında** döngü tespiti (bkz. bölüm 23).
- **Bağlı bileşen (connected components)** sayma/tespit etme problemlerinde.
- **Ağ bağlantı problemleri**: arkadaş grupları (sosyal ağda kim kiminle aynı grupta), dinamik bağlantı sorguları (ör. "şu anda A'dan B'ye bir yol var mı?").
- Görüntü işleme: bağlı bölge (connected component) etiketleme.

### Güçlü yönleri

- İki optimizasyon (path compression + union by rank) birlikte kullanıldığında işlemler **neredeyse sabit zamanlıdır** — çok büyük veri setlerinde bile son derece hızlıdır.
- Implementasyonu şaşırtıcı derecede basittir (birkaç satır kod).

### Zayıf yönleri

- **Kümeleri "bölme" (split/disconnect)** işlemini desteklemez — sadece birleştirme yönünde çalışır; bir bağlantıyı geri almak gerekiyorsa Union-Find uygun değildir.
- Kümenin elemanlarını sıralı/detaylı listelemek gerekiyorsa (sadece "hangi kümede" bilgisini değil, kümenin tüm elemanlarını istiyorsanız) ekstra veri yapısı gerekebilir.

---

## 27. Suffix Tree ve Suffix Array

Bir string'in **tüm sonek (suffix)**'lerini verimli şekilde saklayan yapılardır — string eşleştirme, en uzun tekrar eden alt dizi gibi problemlerde kullanılır.

### Suffix Array (pratikte daha çok tercih edilir)

Bir string'in tüm soneklerinin **sözlük sırasına (lexicographic)** göre sıralanmış index dizisidir.

```python
def suffix_array_olustur(metin):                    # O(n log^2 n) basit implementasyon (n log n de mümkün)
    n = len(metin)
    sonekler = sorted(range(n), key=lambda i: metin[i:])   # O(n^2 log n) naif - basitlik için
    return sonekler

def lcp_dizisi(metin, sa):                            # Longest Common Prefix dizisi - O(n)
    n = len(metin)
    rank = [0] * n
    for i, suf in enumerate(sa):
        rank[suf] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and metin[i + h] == metin[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp

metin = "banana"
sa = suffix_array_olustur(metin)
print(sa)   # [5, 3, 1, 0, 4, 2] -> a, ana, anana, banana, na, nana soneklerinin sıralı index'leri

# Örnek kullanım: bir alt dizinin (substring) metinde geçip geçmediğini binary search ile O(m log n) kontrol
def alt_dizi_var_mi(metin, sa, alt_dizi):
    sol, sag = 0, len(sa) - 1
    while sol <= sag:
        orta = (sol + sag) // 2
        sonek = metin[sa[orta]:sa[orta] + len(alt_dizi)]
        if sonek == alt_dizi:
            return True
        elif sonek < alt_dizi:
            sol = orta + 1
        else:
            sag = orta - 1
    return False

print(alt_dizi_var_mi(metin, sa, "nan"))   # True
```

### Suffix Tree

Tüm sonekleri bir **sıkıştırılmış Trie** (compressed trie) şeklinde saklar — her sonek kökten bir yaprağa giden yoldur. Ukkonen'in algoritmasıyla **O(n)** zamanda inşa edilebilir (implementasyonu oldukça karmaşıktır, bu yüzden burada sadece kavramsal olarak özetliyoruz).

### Karmaşıklık Tablosu

|İşlem|Suffix Array|Suffix Tree|
|---|---|---|
|İnşa|O(n log n) (verimli algoritmalarla)|O(n) (Ukkonen ile)|
|Alt dizi arama|O(m log n)|O(m)|
|Bellek kullanımı|Düşük (sadece index dizisi)|Yüksek (her düğüm + pointer'lar)|
|Implementasyon zorluğu|Orta|Yüksek|

_m = aranan alt dizinin uzunluğu, n = ana metnin uzunluğu_

### Nerede ve ne zaman tercih edilir?

- **Biyoinformatik**: DNA/protein dizisi analizi, genom karşılaştırma.
- **Metin arama motorları**: hızlı alt dizi/pattern arama, en uzun tekrar eden alt dizi bulma.
- **Veri sıkıştırma**: Burrows-Wheeler Transform (bzip2 gibi sıkıştırma algoritmalarının temeli).
- Plagiarism (intihal) tespiti, DNA dizisi hizalama (sequence alignment).

### Güçlü/Zayıf Yönler

- **Suffix Array:** Daha az bellek kullanır ve implementasyonu görece basittir ✅, ama alt dizi araması Suffix Tree'ye göre biraz daha yavaştır (log n çarpanı) ❌.
- **Suffix Tree:** Alt dizi arama O(m) ile çok hızlıdır ✅, ama bellek kullanımı çok yüksektir ve implementasyonu (özellikle O(n) inşa algoritması) oldukça karmaşıktır ❌ — bu yüzden pratikte çoğu sistem Suffix Array + LCP dizisini tercih eder.

---

## 28. Skip List

Sıralı bir bağlı listenin üzerine, rastgele (random) yükseklikte **birden fazla "hızlı yol" katmanı** ekleyerek O(log n) arama/ekleme/silme sağlayan olasılıksal (probabilistic) veri yapısıdır. Dengeli ağaçlara (AVL, Red-Black) rastgelelik tabanlı bir alternatiftir — rotasyon gerektirmez.

```python
import random

class SkipNode:
    def __init__(self, deger, seviye):
        self.deger = deger
        self.ileri = [None] * (seviye + 1)

class SkipList:
    def __init__(self, maks_seviye=16, p=0.5):
        self.maks_seviye = maks_seviye
        self.p = p
        self.head = SkipNode(None, maks_seviye)
        self.seviye = 0

    def _rastgele_seviye(self):
        seviye = 0
        while random.random() < self.p and seviye < self.maks_seviye:
            seviye += 1
        return seviye

    def ekle(self, deger):                          # O(log n) ortalama
        guncelle = [None] * (self.maks_seviye + 1)
        node = self.head
        for i in range(self.seviye, -1, -1):
            while node.ileri[i] and node.ileri[i].deger < deger:
                node = node.ileri[i]
            guncelle[i] = node

        yeni_seviye = self._rastgele_seviye()
        if yeni_seviye > self.seviye:
            for i in range(self.seviye + 1, yeni_seviye + 1):
                guncelle[i] = self.head
            self.seviye = yeni_seviye

        yeni_node = SkipNode(deger, yeni_seviye)
        for i in range(yeni_seviye + 1):
            yeni_node.ileri[i] = guncelle[i].ileri[i]
            guncelle[i].ileri[i] = yeni_node

    def ara(self, deger):                            # O(log n) ortalama
        node = self.head
        for i in range(self.seviye, -1, -1):
            while node.ileri[i] and node.ileri[i].deger < deger:
                node = node.ileri[i]
        node = node.ileri[0]
        return node is not None and node.deger == deger

sl = SkipList()
for deger in [3, 6, 7, 9, 12, 19, 17]:
    sl.ekle(deger)
print(sl.ara(9))    # True
print(sl.ara(100))  # False
```

### Karmaşıklık Tablosu

|İşlem|Ortalama|En Kötü Durum|
|---|---|---|
|Arama|O(log n)|O(n) (çok düşük olasılıkla)|
|Ekleme|O(log n)|O(n)|
|Silme|O(log n)|O(n)|
|Alan|O(n) (ortalama, katmanlar dahil)|O(n log n)|

### Nerede ve ne zaman tercih edilir?

- **Eşzamanlı (concurrent) sistemlerde** — AVL/Red-Black Tree'nin rotasyon sırasında gerektirdiği kilitlemeye (locking) göre skip list'te eşzamanlı erişim yönetimi daha kolaydır (ör. Redis'in **sorted set** yapısı skip list ile implemente edilir).
- Basit ve anlaşılır kod ile dengeli ağaç performansına yakın sonuç istendiğinde.

### Güçlü yönleri

- Dengeli ağaçlara (AVL, Red-Black) göre **implementasyonu çok daha basittir** (rotasyon mantığı yoktur).
- **Eşzamanlılık (concurrency)** açısından dengeli ağaçlardan daha kolay yönetilir — bu yüzden Redis gibi sistemlerde tercih edilir.

### Zayıf yönleri

- **Olasılıksal**dır — garantili O(log n) değil, çok düşük ihtimalle de olsa en kötü durumda O(n)'e düşebilir.
- Dengeli ağaçlara göre biraz daha fazla bellek kullanır (ekstra pointer katmanları).

---

## 29. İndeksleme Teknikleri (ISAM, Linear Indexing, Tree-based Indexing)

Veritabanı ve dosya sistemlerinde, kayıtlara **diski en az kez okuyarak** hızlı erişim sağlamak için kullanılan yöntemlerdir.

### 29.1 Linear Indexing (Doğrusal İndeksleme)

Her indeks girdisi `(anahtar, disk_adresi)` çiftini tutar ve indeks dosyası **sıralı** tutulur; arama, indeks üzerinde **binary search** ile yapılır.

```python
# Basitleştirilmiş linear index simülasyonu
class LinearIndex:
    def __init__(self):
        self.indeks = []      # [(anahtar, veri_konumu), ...] - sıralı tutulur

    def ekle(self, anahtar, konum):                 # O(n) - sıralı ekleme
        import bisect
        bisect.insort(self.indeks, (anahtar, konum), key=lambda x: x[0])

    def ara(self, anahtar):                          # O(log n) - binary search
        import bisect
        i = bisect.bisect_left(self.indeks, anahtar, key=lambda x: x[0])
        if i < len(self.indeks) and self.indeks[i][0] == anahtar:
            return self.indeks[i][1]
        return None

li = LinearIndex()
li.ekle(101, "disk_blok_5")
li.ekle(105, "disk_blok_2")
li.ekle(102, "disk_blok_9")
print(li.ara(102))    # disk_blok_9
```

### 29.2 ISAM (Indexed Sequential Access Method)

Klasik bir veritabanı indeksleme yöntemidir: veriler diskte **sıralı** tutulur, üstüne **statik, çok seviyeli bir indeks** (ör. her N kayıtta bir "işaret" girdisi) inşa edilir. B-Tree'nin atası sayılır ama **statiktir** — indeks bir kere oluşturulduktan sonra yeni eklenen kayıtlar ayrı bir "taşma (overflow) alanına" yazılır ve zamanla performans düşer (periyodik olarak yeniden inşa gerekir).

```python
# ISAM mantığının basitleştirilmiş simülasyonu: iki seviyeli statik indeks
class ISAMBenzeri:
    def __init__(self, veri, blok_boyutu=3):
        self.veri = sorted(veri)                     # ana veri alanı (sıralı)
        self.blok_boyutu = blok_boyutu
        self.indeks = [(self.veri[i][0], i) for i in range(0, len(self.veri), blok_boyutu)]
        self.tasma = []                               # yeni eklenenler buraya gider

    def ara(self, anahtar):                            # O(log(indeks) + blok_boyutu)
        import bisect
        i = bisect.bisect_right(self.indeks, (anahtar, float('inf'))) - 1
        i = max(i, 0)
        baslangic = self.indeks[i][1]
        for j in range(baslangic, min(baslangic + self.blok_boyutu, len(self.veri))):
            if self.veri[j][0] == anahtar:
                return self.veri[j][1]
        for anahtar_t, deger_t in self.tasma:          # taşma alanında da ara
            if anahtar_t == anahtar:
                return deger_t
        return None

    def ekle(self, anahtar, deger):                     # O(1) - taşma alanına eklenir, indeks güncellenmez
        self.tasma.append((anahtar, deger))
```

### 29.3 Tree-based Indexing (Ağaç Tabanlı İndeksleme)

B-Tree/B+ Tree'nin (bkz. bölüm 20) veritabanı indeksi olarak kullanılmasıdır — ISAM'ın aksine **dinamiktir**: yeni kayıtlar eklendiğinde ağaç otomatik olarak yeniden dengelenir (split/merge), performans zamanla bozulmaz.

### Karşılaştırma Tablosu

|Yöntem|Arama|Ekleme|Dinamik mi?|Kullanım|
|---|---|---|---|---|
|Linear Index|O(log n)|O(n) (sıralı ekleme)|Kısmen|Az güncellenen, çoğunlukla okunan veri|
|ISAM|O(log n) + taşma taraması|O(1) (taşmaya) ama zamanla O(n) taşma büyür|Hayır (statik)|Toplu yükleme (batch load) sonrası çok okuma, az yazma|
|Tree-based (B+ Tree)|O(log n)|O(log n)|Evet|Modern veritabanları (sık okuma **ve** yazma)|

### Nerede ve ne zaman tercih edilir?

- **Linear Index:** Veri seyrek güncelleniyor, disk alanı kısıtlı ve basit bir indeks yeterliyse.
- **ISAM:** Veri bir kere toplu yüklenip sonrasında ağırlıklı olarak **okunuyorsa** (az yazma) — ör. arşiv sistemleri, veri ambarı (data warehouse) tarihsel tablolar.
- **Tree-based (B+ Tree):** Modern OLTP veritabanlarının **standart seçimi** — hem sık okuma hem sık yazma olduğunda (MySQL, PostgreSQL, Oracle indeksleri).

### Güçlü/Zayıf Yönler

- **Linear Index:** Basit, az bellek ✅, ama sık ekleme yapılırsa O(n) maliyeti birikir ❌.
- **ISAM:** Toplu okuma-ağırlıklı senaryolarda çok hızlıdır ✅, ama sık yazma ile taşma alanı şişer ve periyodik yeniden inşa (reorganization) gerektirir ❌.
- **Tree-based:** Hem okuma hem yazmada dengeli performans, otomatik dengelenme ✅, ama implementasyon karmaşıklığı en yüksek olanıdır ❌.

---

## 30. Problem Çözme Teknikleri / Pattern'lar

Recursion, Divide & Conquer, Dynamic Programming, Greedy ve Backtracking zaten önceki bölümlerde (10, 13, 14, 15, 16) işlendi. Burada mülakat/rekabetçi programlamada sıkça karşılaşılan **ek pattern'ları** ele alıyoruz.

### 30.1 Brute Force (Kaba Kuvvet)

Tüm olası çözümleri tek tek deneyerek doğru olanı bulma yaklaşımıdır — genelde en basit ama en yavaş çözümdür; diğer tekniklerin (DP, greedy, two pointer vb.) **başlangıç noktası ve doğruluk referansıdır**.

```python
# Örnek: bir dizide toplamı hedefe eşit olan iki eleman bulma - Brute Force O(n^2)
def iki_toplam_brute_force(dizi, hedef):
    n = len(dizi)
    for i in range(n):
        for j in range(i + 1, n):
            if dizi[i] + dizi[j] == hedef:
                return (i, j)
    return None

print(iki_toplam_brute_force([2, 7, 11, 15], 9))   # (0, 1)
```

**Ne zaman tercih edilir?** Girdi boyutu küçükken, hızlıca doğru (ama yavaş) bir çözümle başlayıp optimize etmeden önce test etmek istendiğinde; ya da problem boyutu gerçekten küçükse (n < ~1000) optimize etmeye değmeyebilir. **Güçlü yönü:** Anlaşılması ve doğruluğunu kanıtlaması en kolay yaklaşımdır ✅. **Zayıf yönü:** Genelde üstel/karesel karmaşıklığa sahiptir, büyük veride kullanılamaz ❌.

### 30.2 Randomized Algorithms (Rastgele Algoritmalar)

Çözüm sürecine **rastgelelik** katarak ortalama durumda (expected case) iyi performans elde eden algoritmalardır. İki türü vardır: **Las Vegas** (her zaman doğru sonuç verir, süre değişken — ör. randomized quicksort) ve **Monte Carlo** (süre sabit, doğruluk olasılıksal — ör. Miller-Rabin asallık testi).

```python
import random

# Randomized Quicksort - pivot rastgele seçilir, en kötü durum ihtimalini azaltır
def randomized_quick_sort(dizi):
    if len(dizi) <= 1:
        return dizi
    pivot = random.choice(dizi)
    kucuk = [x for x in dizi if x < pivot]
    esit = [x for x in dizi if x == pivot]
    buyuk = [x for x in dizi if x > pivot]
    return randomized_quick_sort(kucuk) + esit + randomized_quick_sort(buyuk)

print(randomized_quick_sort([5, 2, 8, 1, 9, 3]))   # [1, 2, 3, 5, 8, 9]

# Örnek: bir dizide rastgele örnekleme ile yaklaşık medyan tahmini (Monte Carlo mantığı)
def yaklasik_medyan(dizi, ornek_boyutu=100):
    ornek = random.sample(dizi, min(ornek_boyutu, len(dizi)))
    ornek.sort()
    return ornek[len(ornek) // 2]
```

**Ne zaman tercih edilir?** Deterministik algoritmanın en kötü durum performansı kötüyse (ör. quicksort'un sıralı veride O(n²) riski) ve rastgelelikle bu riski **istatistiksel olarak** azaltmak mümkünse; büyük veri setlerinde yaklaşık/örnekleme tabanlı çözümler yeterliyse. **Güçlü yönü:** En kötü durum senaryolarına karşı **beklenen (expected)** performansı iyileştirir, bazı problemleri deterministik çözümlerden çok daha basit hale getirir ✅. **Zayıf yönü:** Sonuç garantisi yoktur (Monte Carlo'da hata payı vardır) veya çalışma süresi garanti edilemez (Las Vegas'ta) ❌.

### 30.3 Kth Element (K'ıncı Eleman)

Bir dizideki **k'ıncı en küçük/büyük** elemanı, tam sıralama yapmadan bulma problemidir. **Quickselect** (quicksort'un partition mantığından türetilir) ortalama O(n)'de çözer; heap ile O(n log k) alternatif de vardır.

```python
import random

def quickselect(dizi, k):                      # k'ıncı en küçük eleman (0-indexed) - ortalama O(n)
    if len(dizi) == 1:
        return dizi[0]
    pivot = random.choice(dizi)
    kucuk = [x for x in dizi if x < pivot]
    esit = [x for x in dizi if x == pivot]
    buyuk = [x for x in dizi if x > pivot]

    if k < len(kucuk):
        return quickselect(kucuk, k)
    elif k < len(kucuk) + len(esit):
        return pivot
    else:
        return quickselect(buyuk, k - len(kucuk) - len(esit))

print(quickselect([7, 10, 4, 3, 20, 15], 2))   # 3. en küçük eleman (0-indexed k=2) -> 7

# Heap ile "en büyük k eleman" - O(n log k) - bkz. bölüm 8
import heapq
def k_en_buyuk(dizi, k):
    return heapq.nlargest(k, dizi)
```

**Ne zaman tercih edilir?** Medyan bulma, "en büyük/küçük k eleman" problemleri, tüm diziyi sıralamaya gerek olmadığında (sıralama O(n log n)'e karşı quickselect ortalama O(n)). **Güçlü yönü:** Tam sıralamaya göre çok daha hızlıdır (ortalama O(n)) ✅. **Zayıf yönü:** En kötü durumda O(n²)'ye düşebilir (kötü pivot seçimi — quicksort ile aynı risk) ❌.

### 30.4 Two Pointer Technique (İki İşaretçi)

Genellikle **sıralı** bir dizide, iki index'i (baştan ve/veya sondan) aynı anda hareket ettirerek O(n²) brute force'u O(n)'e indirger.

```python
# İki toplam problemi - sıralı dizide two pointer ile O(n)
def iki_toplam_sirali(dizi, hedef):
    sol, sag = 0, len(dizi) - 1
    while sol < sag:
        toplam = dizi[sol] + dizi[sag]
        if toplam == hedef:
            return (sol, sag)
        elif toplam < hedef:
            sol += 1
        else:
            sag -= 1
    return None

print(iki_toplam_sirali([2, 7, 11, 15], 9))   # (0, 1)

# Palindrom kontrolü - O(n)
def palindrom_mu(s):
    sol, sag = 0, len(s) - 1
    while sol < sag:
        if s[sol] != s[sag]:
            return False
        sol += 1
        sag -= 1
    return True
```

**Ne zaman tercih edilir?** Sıralı dizi/string üzerinde çift eleman ilişkisi (toplam, fark) aranırken, veya bir diziyi baştan ve sondan aynı anda taramak gerektiğinde (palindrom, konteyner su problemi). **Güçlü/Zayıf:** O(n²)'yi O(n)'e indirger, ekstra bellek gerekmez ✅; ama genelde **sıralı veri** gerektirir (sırasızsa önce sıralama O(n log n) maliyeti eklenir) ❌.

### 30.5 Sliding Window Technique (Kayan Pencere)

Bir dizi/string üzerinde **sabit veya değişken boyutlu bir "pencere"**yi kaydırarak, her adımda pencereyi baştan yeniden hesaplamak yerine sadece giren/çıkan elemanı güncelleyerek O(n)'de çözer.

```python
# Sabit boyutlu pencere: k uzunluğundaki en büyük toplam alt dizi - O(n)
def maks_alt_dizi_toplami(dizi, k):
    pencere_toplami = sum(dizi[:k])
    maks_toplam = pencere_toplami
    for i in range(k, len(dizi)):
        pencere_toplami += dizi[i] - dizi[i - k]   # yeni gireni ekle, eskiyi çıkar
        maks_toplam = max(maks_toplam, pencere_toplami)
    return maks_toplam

print(maks_alt_dizi_toplami([2, 1, 5, 1, 3, 2], 3))   # 9  (5+1+3)

# Değişken boyutlu pencere: en küçük toplamı >= hedef olan alt dizinin uzunluğu - O(n)
def en_kucuk_alt_dizi_uzunlugu(dizi, hedef):
    sol, toplam, min_uzunluk = 0, 0, float('inf')
    for sag in range(len(dizi)):
        toplam += dizi[sag]
        while toplam >= hedef:
            min_uzunluk = min(min_uzunluk, sag - sol + 1)
            toplam -= dizi[sol]
            sol += 1
    return min_uzunluk if min_uzunluk != float('inf') else 0

print(en_kucuk_alt_dizi_uzunlugu([2, 1, 5, 2, 3, 2], 7))   # 2 (5+2)
```

**Ne zaman tercih edilir?** "Ardışık alt dizi/alt string" üzerinde toplam, maksimum, en uzun/kısa benzersiz alt dizi gibi problemlerde (ör. "tekrar eden karaktersiz en uzun alt string"). **Güçlü/Zayıf:** Brute force'un O(n·k) veya O(n²)'sini O(n)'e indirger ✅; ama sadece **ardışık (contiguous)** alt dizi/alt string problemlerinde işe yarar, ardışık olmayan alt kümeler için uygun değildir ❌.

### 30.6 Fast and Slow Pointers (Tavşan-Kaplumbağa)

İki işaretçi, biri **1 adım (slow)**, diğeri **2 adım (fast)** ilerler. Genelde bağlı liste problemlerinde döngü tespiti ve orta eleman bulma için kullanılır (Floyd's Cycle Detection).

```python
class Node:
    def __init__(self, deger):
        self.deger = deger
        self.sonraki = None

def dongu_var_mi(head):                          # O(n) zaman, O(1) alan
    yavas = hizli = head
    while hizli and hizli.sonraki:
        yavas = yavas.sonraki
        hizli = hizli.sonraki.sonraki
        if yavas == hizli:
            return True
    return False

def ortadaki_dugum(head):                          # O(n) - tek geçişte orta eleman
    yavas = hizli = head
    while hizli and hizli.sonraki:
        yavas = yavas.sonraki
        hizli = hizli.sonraki.sonraki
    return yavas

# Test: döngülü liste
a, b, c = Node(1), Node(2), Node(3)
a.sonraki, b.sonraki, c.sonraki = b, c, a   # döngü oluştur
print(dongu_var_mi(a))    # True
```

**Ne zaman tercih edilir?** Bağlı listede döngü tespiti, orta elemanı bulma, listenin bir palindrom olup olmadığını kontrol etme — ekstra bellek (hash set) kullanmadan. **Güçlü/Zayıf:** O(1) ekstra bellek ile çözer (hash set kullanan alternatife göre çok daha az bellek) ✅; ama sadece belirli problem türlerine (döngüsel/doğrusal yapılar) özgüdür ❌.

### 30.7 Merge Intervals (Aralıkları Birleştirme)

Örtüşen (overlapping) aralıkları birleştirme problemidir. Önce başlangıca göre sırala, sonra ardışık aralıkları kontrol ederek birleştir.

```python
def araliklari_birlestir(araliklar):              # O(n log n) - sıralama baskın maliyet
    if not araliklar:
        return []
    araliklar = sorted(araliklar, key=lambda x: x[0])
    birlesmis = [araliklar[0]]
    for baslangic, bitis in araliklar[1:]:
        son_baslangic, son_bitis = birlesmis[-1]
        if baslangic <= son_bitis:                  # örtüşüyor
            birlesmis[-1] = (son_baslangic, max(son_bitis, bitis))
        else:
            birlesmis.append((baslangic, bitis))
    return birlesmis

print(araliklari_birlestir([(1,3), (2,6), (8,10), (15,18)]))
# [(1, 6), (8, 10), (15, 18)]
```

**Ne zaman tercih edilir?** Takvim/toplantı çakışma tespiti, zaman aralığı birleştirme, kaynak rezervasyon sistemleri. **Güçlü/Zayıf:** Sezgisel ve uygulaması kolaydır ✅; sıralama gerektirdiği için O(n log n) alt sınırı vardır (aralıklar zaten sıralıysa O(n)'e iner) ❌.

### 30.8 Cyclic Sort (Döngüsel Sıralama)

`1..n` aralığındaki (veya benzer sınırlı aralıktaki) sayıları içeren bir dizide, her elemanı **kendi doğru index'ine** yerleştirerek O(n) zamanda, O(1) ekstra alanla sıralama/eksik-tekrar eden eleman bulma yapar.

```python
def cyclic_sort(dizi):                             # O(n) zaman, O(1) alan - dizi [1..n] içermeli
    i = 0
    while i < len(dizi):
        dogru_index = dizi[i] - 1
        if dizi[i] != dizi[dogru_index]:
            dizi[i], dizi[dogru_index] = dizi[dogru_index], dizi[i]
        else:
            i += 1
    return dizi

print(cyclic_sort([3, 1, 5, 4, 2]))    # [1, 2, 3, 4, 5]

# Örnek kullanım: 1..n aralığında eksik sayıyı bulma - O(n)
def eksik_sayi(dizi):
    cyclic_sort(dizi)
    for i, deger in enumerate(dizi):
        if deger != i + 1:
            return i + 1
    return len(dizi) + 1
```

**Ne zaman tercih edilir?** Veri, `1..n` gibi **bilinen ve sınırlı bir aralıkta** olduğunda — eksik/tekrar eden sayı bulma, yerinde sıralama problemlerinde (genel sıralama algoritmalarına göre çok daha hızlı). **Güçlü/Zayıf:** O(n) zaman + O(1) alan ile çalışır (genel sıralama algoritmalarından çok daha hızlı) ✅; ama sadece **sınırlı, bilinen aralıktaki** tam sayı dizilerinde işe yarar, genel amaçlı değildir ❌.

### 30.9 Two Heaps (İki Yığın)

Bir veri akışının (data stream) **medyanını** anlık olarak takip etmek için iki heap kullanılır: küçük yarı için **max-heap**, büyük yarı için **min-heap**. İki heap'in boyutu her zaman dengede (fark en fazla 1) tutulur.

```python
import heapq

class MedyanBulucu:
    def __init__(self):
        self.kucuk_yari = []     # max-heap (Python'da negatiflenerek)
        self.buyuk_yari = []     # min-heap

    def sayi_ekle(self, sayi):                     # O(log n)
        heapq.heappush(self.kucuk_yari, -sayi)
        heapq.heappush(self.buyuk_yari, -heapq.heappop(self.kucuk_yari))
        if len(self.buyuk_yari) > len(self.kucuk_yari):
            heapq.heappush(self.kucuk_yari, -heapq.heappop(self.buyuk_yari))

    def medyan_bul(self):                            # O(1)
        if len(self.kucuk_yari) > len(self.buyuk_yari):
            return -self.kucuk_yari[0]
        return (-self.kucuk_yari[0] + self.buyuk_yari[0]) / 2

mb = MedyanBulucu()
for sayi in [5, 15, 1, 3]:
    mb.sayi_ekle(sayi)
    print(mb.medyan_bul())    # 5, 10.0, 5, 4.0
```

**Ne zaman tercih edilir?** Sürekli akan veride (stream) anlık medyan/orta değer takibi gerektiğinde — tüm veriyi her seferinde yeniden sıralamak yerine. **Güçlü/Zayıf:** Her yeni eleman O(log n)'de eklenir ve medyan O(1)'de okunur (tüm veriyi sıralamaktan çok daha verimli) ✅; ama sadece medyan/orta değer gibi belirli istatistikler için tasarlanmıştır, genel amaçlı değildir ❌.

### 30.10 Island Traversal (Ada Gezinme / Matris DFS-BFS)

2D bir matriste (ör. harita, görüntü) birbirine bağlı hücre gruplarını ("adalar") saymak veya keşfetmek için DFS/BFS'in matrise uygulanmasıdır.

```python
def ada_sayisi(izgara):                            # O(satir * sutun)
    if not izgara:
        return 0
    satirlar, sutunlar = len(izgara), len(izgara[0])
    ziyaret = set()

    def dfs(r, c):
        if (r < 0 or r >= satirlar or c < 0 or c >= sutunlar or
            izgara[r][c] == 0 or (r, c) in ziyaret):
            return
        ziyaret.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:    # 4 yön
            dfs(r + dr, c + dc)

    sayac = 0
    for r in range(satirlar):
        for c in range(sutunlar):
            if izgara[r][c] == 1 and (r, c) not in ziyaret:
                dfs(r, c)
                sayac += 1
    return sayac

izgara = [
    [1, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
]
print(ada_sayisi(izgara))   # 3
```

**Ne zaman tercih edilir?** Görüntü işleme (bağlı bölge etiketleme), harita/oyun ızgaralarında bölge sayma, kelime bulma oyunları (Boggle — komşu hücrelerde kelime arama). **Güçlü/Zayıf:** Graf traversal mantığını (DFS/BFS) doğrudan 2D matrise uygulayarak sezgisel bir çözüm sunar ✅; recursive DFS büyük matrislerde call stack limiti riski taşır — bu durumda iteratif (stack tabanlı) versiyon tercih edilmelidir ❌.

### 30.11 Multi-threaded Yaklaşım (Çoklu İş Parçacığı)

Bir problemi **bağımsız alt görevlere** bölüp paralel thread/process'lerde çalıştırarak toplam süreyi azaltma yaklaşımıdır. Python'da GIL (Global Interpreter Lock) nedeniyle CPU-bound işlerde gerçek paralellik için `multiprocessing`, I/O-bound işlerde `threading` tercih edilir.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O-bound örnek: birden fazla "iş" paralel thread'lerde (ör. ağ isteği simülasyonu)
def veri_getir(id):
    return f"veri-{id} getirildi"

with ThreadPoolExecutor(max_workers=4) as executor:
    sonuclar = list(executor.map(veri_getir, range(4)))
print(sonuclar)

# CPU-bound örnek: büyük bir diziyi parçalara bölüp paralel işleme (multiprocessing)
def kare_topla(parca):
    return sum(x * x for x in parca)

def paralel_kare_toplami(dizi, islemci_sayisi=4):     # O(n / p) - p işlemci ile
    parca_boyutu = len(dizi) // islemci_sayisi
    parcalar = [dizi[i:i+parca_boyutu] for i in range(0, len(dizi), parca_boyutu)]
    with ProcessPoolExecutor(max_workers=islemci_sayisi) as executor:
        sonuclar = executor.map(kare_topla, parcalar)
    return sum(sonuclar)

print(paralel_kare_toplami(list(range(1, 21)), islemci_sayisi=4))
```

**Ne zaman tercih edilir?** Büyük ve **bağımsız alt görevlere bölünebilen** işlerde (ör. büyük veri işleme, toplu görüntü/video işleme, çoklu ağ isteği). CPU-bound işlerde `multiprocessing`, I/O-bound (ağ, disk bekleme) işlerde `threading`/`asyncio` tercih edilir. **Güçlü/Zayıf:** Doğru kullanıldığında toplam süreyi işlemci/çekirdek sayısına yakın oranda azaltır ✅; senkronizasyon (race condition, deadlock riski), Python'da GIL kısıtı (threading'de gerçek CPU paralelliği yoktur) ve debug zorluğu gibi ek karmaşıklıklar getirir ❌.

> **Not:** Listendeki **Selection Sort** (bkz. bölüm 12.2), **BFS/DFS** (bkz. bölüm 9 ve 21), **Greedy Algorithms** (bkz. bölüm 15), **Backtracking** (bkz. bölüm 16), **Divide and Conquer** (bkz. bölüm 13), **Recursion** (bkz. bölüm 10) ve **Dynamic Programming** (bkz. bölüm 14) zaten notların önceki sürümünde mevcuttu, bu yüzden tekrar eklenmedi.

## 31. Özet Karşılaştırma Tabloları

### Veri Yapıları — Genel Bakış

|Veri Yapısı|Erişim|Arama|Ekleme|Silme|En Uygun Kullanım|
|---|---|---|---|---|---|
|Array (dizi)|O(1)|O(n)|O(n)*|O(n)|Rastgele erişim, sabit/az değişen veri|
|Linked List|O(n)|O(n)|O(1)**|O(1)**|Sık başa/ortaya ekleme-silme|
|Stack|O(n)|O(n)|O(1)|O(1)|LIFO — undo, call stack, DFS|
|Queue (deque)|O(n)|O(n)|O(1)|O(1)|FIFO — BFS, iş kuyruğu|
|Hash Table|O(1) ort.|O(1) ort.|O(1) ort.|O(1) ort.|Hızlı arama/varlık kontrolü|
|BST (dengeli)|O(log n)|O(log n)|O(log n)|O(log n)|Sıralı veri + hızlı işlem|
|Heap|O(1) (min/max)|O(n)|O(log n)|O(log n)|Öncelik kuyruğu, en iyi-k|
|Graph (liste)|—|O(V+E)|O(1)|O(V)|İlişkisel/ağ verisi|

*sonuna ekleme amortized O(1); **node referansı biliniyorsa

### Algoritma Aileleri — Ne Zaman Hangisi?

|Teknik|Ana Fikir|Tipik Karmaşıklık|Örnek Problemler|
|---|---|---|---|
|Divide & Conquer|Böl, çöz, birleştir|O(n log n)|Merge/Quick Sort, hızlı üs|
|Dynamic Programming|Örtüşen alt problemleri sakla|O(n), O(n²)|Knapsack, LCS, Fibonacci|
|Greedy|Her adımda yerel en iyiyi seç|O(n log n)|MST, aktivite seçme, Dijkstra|
|Backtracking|Dene, geçersizse geri dön|O(2ⁿ), O(n!)|N-Queens, Sudoku, permütasyon|
|Recursion|Kendini çağırma|Probleme bağlı|Ağaç/graf gezinme|

### Hızlı Karar Rehberi

- **Hızlı arama/ekleme mi lazım, sıralama önemli değil mi?** → Hash Table
- **Sıralı veri + hem arama hem ekleme hızlı olsun mu?** → Dengeli BST
- **Sadece en küçük/büyüğe erişim yeterli mi?** → Heap
- **LIFO davranış mı gerekiyor?** → Stack
- **FIFO davranış mı gerekiyor?** → Queue (deque)
- **İlişkisel/ağ verisi mi?** → Graph + BFS/DFS
- **Aynı alt problem tekrar tekrar çözülüyor mu?** → Dynamic Programming
- **Yerel en iyi seçim global çözüme götürüyor mu (kanıtlanabilir)?** → Greedy
- **Tüm olası çözümleri denemek mi gerekiyor?** → Backtracking