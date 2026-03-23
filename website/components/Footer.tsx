import { Badge } from "@/components/ui/badge";

export function Footer() {
  return (
    <footer className="mx-auto mt-20 w-[min(1100px,95vw)] rounded-2xl border border-black/10 bg-white/70 p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <p className="text-sm text-brand-slate">Built for cognitive accessibility in AI systems.</p>
        <div className="flex flex-wrap gap-2">
          <a href="https://github.com" target="_blank" rel="noreferrer"><Badge>GitHub</Badge></a>
          <a href="https://discord.com" target="_blank" rel="noreferrer"><Badge>Discord</Badge></a>
          <a href="https://npmjs.com" target="_blank" rel="noreferrer"><Badge>npm</Badge></a>
          <a href="https://pypi.org" target="_blank" rel="noreferrer"><Badge>PyPI</Badge></a>
        </div>
      </div>
    </footer>
  );
}
