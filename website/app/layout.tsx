import type { Metadata } from "next";

import { Footer } from "@/components/Footer";
import { Nav } from "@/components/Nav";

import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroBridge",
  description: "AI accessibility layer for neurodivergent communication"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="mx-auto w-[min(1100px,95vw)] py-8">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
