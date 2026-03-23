export type AdaptRequest = {
  text: string;
  profile: "adhd" | "autism" | "dyslexia" | "anxiety" | "dyscalculia";
  output_format?: "markdown" | "plain" | "html" | "json";
  user_id?: string;
};

export type AdaptResponse = {
  adapted_text: string;
  raw_text: string;
  modules_run: string[];
  processing_ms: number;
  profile: string;
};

export async function adaptText(input: AdaptRequest): Promise<AdaptResponse> {
  const response = await fetch("/api/adapt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input)
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload?.error ?? "Adaptation failed");
  }

  return response.json();
}

export async function submitQuiz(answers: number[]) {
  const response = await fetch("/api/quiz", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers })
  });

  if (!response.ok) {
    throw new Error("Quiz submission failed");
  }

  return response.json();
}
