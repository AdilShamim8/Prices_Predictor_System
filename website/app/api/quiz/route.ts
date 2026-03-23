import { NextRequest, NextResponse } from "next/server";

const profileMap = ["adhd", "autism", "dyslexia", "anxiety", "dyscalculia"];

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => ({}))) as { answers?: number[] };
  const answers = Array.isArray(body.answers) ? body.answers : [];

  if (!answers.length) {
    return NextResponse.json({ error: "answers are required" }, { status: 400 });
  }

  const totals = profileMap.map((_, index) =>
    answers.reduce((sum, score, answerIndex) => sum + (answerIndex % 5 === index ? Number(score || 0) : 0), 0)
  );

  const primaryIndex = totals.indexOf(Math.max(...totals));
  return NextResponse.json({
    primary_profile: profileMap[primaryIndex] ?? "adhd",
    confidence: 0.72,
    scores: Object.fromEntries(profileMap.map((key, index) => [key, totals[index]]))
  });
}
