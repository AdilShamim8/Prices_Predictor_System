"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { submitQuiz } from "@/lib/api";

export default function QuizPage() {
  const [answers, setAnswers] = useState<number[]>(Array(5).fill(0));
  const [result, setResult] = useState<string>("");

  async function handleSubmit() {
    const payload = await submitQuiz(answers);
    setResult(payload.primary_profile ?? "profile unavailable");
  }

  return (
    <section className="space-y-5">
      <h1 className="text-4xl font-black text-brand-ink">ProfileQuiz</h1>
      <p className="text-brand-slate">Quick profile estimation preview. Full guided quiz arrives in Day 17.</p>
      <div className="rounded-2xl border border-black/10 bg-white/80 p-6">
        <p className="text-sm text-brand-slate">Sample controls:</p>
        <div className="mt-3 grid gap-2 sm:grid-cols-5">
          {answers.map((value, index) => (
            <input
              key={index}
              type="range"
              min={0}
              max={4}
              value={value}
              onChange={(event) => {
                const next = [...answers];
                next[index] = Number(event.target.value);
                setAnswers(next);
              }}
            />
          ))}
        </div>
        <Button className="mt-4" onClick={handleSubmit}>Calculate profile</Button>
        {result ? <p className="mt-3 font-semibold text-brand-primary">Suggested profile: {result}</p> : null}
      </div>
    </section>
  );
}
