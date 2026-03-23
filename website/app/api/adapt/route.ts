import { NextRequest, NextResponse } from "next/server";

type AdaptBody = {
  text?: string;
  profile?: string;
  output_format?: string;
  user_id?: string;
};

const limiter = new Map<string, { count: number; resetAt: number }>();

function getIp(req: NextRequest): string {
  const forwarded = req.headers.get("x-forwarded-for");
  if (forwarded) {
    return forwarded.split(",")[0]?.trim() || "unknown";
  }
  return "unknown";
}

function rateLimit(ip: string): boolean {
  const now = Date.now();
  const current = limiter.get(ip);
  if (!current || now > current.resetAt) {
    limiter.set(ip, { count: 1, resetAt: now + 60_000 });
    return true;
  }
  if (current.count >= 10) {
    return false;
  }
  current.count += 1;
  return true;
}

function mockAdapt(text: string, profile: string) {
  const softened = text
    .replace(/\bASAP\b/gi, "when you have time")
    .replace(/\bimmediately\b/gi, "when ready")
    .replace(/\bcritical\b/gi, "important");

  const chunks = softened
    .split(/(?<=[.!?])\s+/)
    .reduce<string[]>((acc, sentence, index) => {
      const slot = Math.floor(index / 2);
      if (!acc[slot]) {
        acc[slot] = sentence;
      } else {
        acc[slot] = `${acc[slot]} ${sentence}`;
      }
      return acc;
    }, [])
    .join("\n\n");

  return {
    raw_text: text,
    adapted_text: chunks,
    modules_run: ["Chunker", profile === "anxiety" ? "UrgencyFilter" : "ToneRewriter"],
    processing_ms: 6,
    profile
  };
}

export async function POST(req: NextRequest) {
  const ip = getIp(req);
  if (!rateLimit(ip)) {
    return NextResponse.json({ error: "Rate limit exceeded. Try again in a minute." }, { status: 429 });
  }

  const payload = (await req.json()) as AdaptBody;
  const text = payload.text?.trim() || "";
  const profile = (payload.profile || "adhd").toLowerCase();

  if (!text) {
    return NextResponse.json({ error: "text is required" }, { status: 400 });
  }

  const backendUrl = process.env.NEUROBRIDGE_API_URL;
  if (!backendUrl) {
    return NextResponse.json(mockAdapt(text, profile));
  }

  try {
    const response = await fetch(`${backendUrl.replace(/\/$/, "")}/api/v1/adapt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        profile,
        output_format: payload.output_format || "markdown",
        user_id: payload.user_id
      })
    });

    const result = await response.json();
    if (!response.ok) {
      return NextResponse.json({ error: result?.detail || "Backend adaptation failed" }, { status: response.status });
    }
    return NextResponse.json(result);
  } catch {
    return NextResponse.json({ error: "Unable to reach NeuroBridge API." }, { status: 502 });
  }
}
