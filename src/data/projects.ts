// Portfolyo girdileri. Sayfayi doldurmak icin asagidaki diziye eleman ekle;
// dizi bosken /projects sayfasi yalnizca yerel gelistirmede uyari gosterir.
//
// Her girdinin barajı su uc alan: problem nedir, hangi karari neden verdin,
// ve bunu dogrulayan somut bir sey (repo, bir sayi, bir grafik). Ucu de
// dolmayan bir proje listelenmemeli — zayif girdi, girdi olmamasindan kotudur.

export type ProjectStatus = "aktif" | "tamamlandi" | "planlanan";

export type Project = {
	/** Projenin adi. */
	title: string;
	status: ProjectStatus;
	/** Hangi problemi cozuyor — tek cumle, ozellik listesi degil. */
	problem: string;
	/** Aldigin en ilginc teknik karar ve gerekcesi. */
	decision: string;
	/** Dogrulanabilir sonuc: bir sayi, bir olcum, bir grafik. */
	evidence: string;
	/** Kullanilan araclar. */
	stack: string[];
	/** Kaynak kod baglantisi. */
	repo?: string;
	/** Bu blogdaki ilgili yazi, orn. "/posts/yz50-12-backpropagation-calculus/". */
	post?: string;
	/** Bitis ya da son guncelleme tarihi, orn. "2026-10". */
	date?: string;
};

export const projects: Project[] = [
	// Ornek girdi (kopyalayip doldur, yorumu kaldir):
	//
	// {
	// 	title: "Ledger API",
	// 	status: "aktif",
	// 	problem: "...",
	// 	decision: "...",
	// 	evidence: "...",
	// 	stack: ["FastAPI", "PostgreSQL"],
	// 	repo: "https://github.com/CoYildiz/ledger_api",
	// 	date: "2026-10",
	// },
];

export const statusLabels: Record<ProjectStatus, string> = {
	aktif: "Devam ediyor",
	tamamlandi: "Tamamlandı",
	planlanan: "Planlanan",
};
