import { type CollectionEntry, getCollection } from "astro:content";
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import { getCategoryUrl } from "@utils/url-utils.ts";

// // Retrieve posts and sort them by publication date
async function getRawSortedPosts() {
	const allBlogPosts = await getCollection("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});

	const sorted = allBlogPosts.sort((a, b) => {
		const dateA = new Date(a.data.published);
		const dateB = new Date(b.data.published);
		return dateA > dateB ? -1 : 1;
	});
	return sorted;
}

// Seri kategorisi: bu kategorideki yazilar ana akistan ve arsivden cikarilir,
// kendi seri sayfalarinda (/ml) numara sirasiyla listelenir.
export const SERIES_CATEGORY = "ML";

function isSeriesPost(data: { category?: string | null }): boolean {
	return (data.category ?? "").trim() === SERIES_CATEGORY;
}

// Slug'daki iki haneli sira numarasi: "yz50-07-neden-..." -> 7
function seriesOrder(slug: string): number {
	const m = slug.match(/(?:^|-)(\d{2})-/);
	return m ? Number.parseInt(m[1], 10) : Number.MAX_SAFE_INTEGER;
}

// Ileri/geri gezinme baglantilarini verilen liste icinde zincirler.
function linkNeighbours(list: CollectionEntry<"posts">[]) {
	for (let i = 1; i < list.length; i++) {
		list[i].data.nextSlug = list[i - 1].slug;
		list[i].data.nextTitle = list[i - 1].data.title;
	}
	for (let i = 0; i < list.length - 1; i++) {
		list[i].data.prevSlug = list[i + 1].slug;
		list[i].data.prevTitle = list[i + 1].data.title;
	}
	return list;
}

// Ana akis ve RSS: seri yazilari haric, tarihe gore yeniden eskiye.
export async function getSortedPosts() {
	const sorted = (await getRawSortedPosts()).filter(
		(post) => !isSeriesPost(post.data),
	);
	return linkNeighbours(sorted);
}

// Seri yazilari, numara sirasiyla (01 -> 24).
export async function getSeriesPosts() {
	const series = (await getRawSortedPosts())
		.filter((post) => isSeriesPost(post.data))
		.sort((a, b) => seriesOrder(a.slug) - seriesOrder(b.slug));

	// Seri icinde "onceki" bir kucuk numara, "sonraki" bir buyuk numara olmali.
	// linkNeighbours listeyi yeniden-eskiye varsaydigi icin ters cevirip veriyoruz.
	linkNeighbours([...series].reverse());
	return series;
}

// Butun yazilar: sayfa uretimi icin (her ikisi de kendi zincirini korur).
export async function getAllPostsForRouting() {
	const [blog, series] = await Promise.all([getSortedPosts(), getSeriesPosts()]);
	return [...blog, ...series];
}
export type PostForList = {
	slug: string;
	data: CollectionEntry<"posts">["data"];
};
export async function getSortedPostsList(): Promise<PostForList[]> {
	const sortedFullPosts = await getRawSortedPosts();

	// delete post.body
	const sortedPostsList = sortedFullPosts.map((post) => ({
		slug: post.slug,
		data: post.data,
	}));

	return sortedPostsList;
}
export type Tag = {
	name: string;
	count: number;
};

export async function getTagList(): Promise<Tag[]> {
	const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});

	const countMap: { [key: string]: number } = {};
	allBlogPosts.forEach((post: { data: { tags: string[] } }) => {
		post.data.tags.forEach((tag: string) => {
			if (!countMap[tag]) countMap[tag] = 0;
			countMap[tag]++;
		});
	});

	// sort tags
	const keys: string[] = Object.keys(countMap).sort((a, b) => {
		return a.toLowerCase().localeCompare(b.toLowerCase());
	});

	return keys.map((key) => ({ name: key, count: countMap[key] }));
}

export type Category = {
	name: string;
	count: number;
	url: string;
};

export async function getCategoryList(): Promise<Category[]> {
	const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});
	const count: { [key: string]: number } = {};
	allBlogPosts.forEach((post: { data: { category: string | null } }) => {
		if (!post.data.category) {
			const ucKey = i18n(I18nKey.uncategorized);
			count[ucKey] = count[ucKey] ? count[ucKey] + 1 : 1;
			return;
		}

		const categoryName =
			typeof post.data.category === "string"
				? post.data.category.trim()
				: String(post.data.category).trim();

		count[categoryName] = count[categoryName] ? count[categoryName] + 1 : 1;
	});

	const lst = Object.keys(count).sort((a, b) => {
		return a.toLowerCase().localeCompare(b.toLowerCase());
	});

	const ret: Category[] = [];
	for (const c of lst) {
		ret.push({
			name: c,
			count: count[c],
			url: getCategoryUrl(c),
		});
	}
	return ret;
}
