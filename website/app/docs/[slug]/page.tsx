import fs from "node:fs/promises";
import path from "node:path";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";

type Props = {
  params: { slug: string };
};

const docsDir = path.join(process.cwd(), "content", "docs");

export default async function DocPage({ params }: Props) {
  const filePath = path.join(docsDir, `${params.slug}.mdx`);
  const fallbackPath = path.join(docsDir, `${params.slug}.md`);

  let source = "";
  try {
    source = await fs.readFile(filePath, "utf8");
  } catch {
    try {
      source = await fs.readFile(fallbackPath, "utf8");
    } catch {
      notFound();
    }
  }

  return (
    <article className="prose prose-slate max-w-3xl rounded-2xl bg-white/80 p-8">
      <MDXRemote source={source} />
    </article>
  );
}
