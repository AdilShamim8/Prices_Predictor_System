import fs from "node:fs/promises";
import path from "node:path";
import Link from "next/link";

const docsDir = path.join(process.cwd(), "content", "docs");

export default async function DocsIndexPage() {
  const entries = await fs.readdir(docsDir, { withFileTypes: true });
  const docs = entries
    .filter((entry) => entry.isFile() && (entry.name.endsWith(".md") || entry.name.endsWith(".mdx")))
    .map((entry) => ({
      slug: entry.name.replace(/\.mdx?$/, ""),
      name: entry.name.replace(/\.mdx?$/, "").replace(/-/g, " ")
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return (
    <section>
      <h1 className="text-4xl font-black text-brand-ink">Documentation</h1>
      <p className="mt-2 text-brand-slate">Browse guides and examples written in MDX.</p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        {docs.map((doc) => (
          <Link
            key={doc.slug}
            href={`/docs/${doc.slug}`}
            className="rounded-xl border border-black/10 bg-white/80 p-4 text-sm capitalize hover:border-brand-primary/40"
          >
            {doc.name}
          </Link>
        ))}
      </div>
    </section>
  );
}
