import { connection } from "next/server";
import { Suspense } from "react";

type Readiness = {
  status: "ready" | "not_ready";
  database: boolean;
  pgvector_version: string | null;
};

async function getReadiness(): Promise<Readiness | null> {
  // Inside Docker the server reaches the API by service name, not localhost.
  const apiUrl =
    process.env.API_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_API_URL ??
    "http://localhost:8000";

  try {
    const response = await fetch(`${apiUrl}/health/ready`, {
      signal: AbortSignal.timeout(5000),
    });
    // A 503 still carries a JSON body saying which part is down.
    return (await response.json()) as Readiness;
  } catch {
    return null;
  }
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-8 py-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}

async function SystemStatus() {
  await connection(); // Always check live, never at build time.
  const readiness = await getReadiness();

  return (
    <dl className="divide-y text-sm">
      <StatusRow label="API" value={readiness ? "Reachable" : "Unreachable"} />
      <StatusRow
        label="Database"
        value={readiness?.database ? "Connected" : "Unavailable"}
      />
      <StatusRow
        label="pgvector"
        value={readiness?.pgvector_version ?? "Unavailable"}
      />
    </dl>
  );
}

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-md flex-1 px-4 py-16">
      <h1 className="text-2xl font-semibold tracking-tight">
        Campus AI Assistant
      </h1>
      <p className="text-muted-foreground mt-2 mb-8 text-sm">
        Project scaffold. The chat interface arrives in Milestone 4.
      </p>
      <Suspense
        fallback={
          <p className="text-muted-foreground text-sm">Checking system…</p>
        }
      >
        <SystemStatus />
      </Suspense>
    </main>
  );
}
