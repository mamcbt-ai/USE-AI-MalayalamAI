import { NextResponse } from "next/server";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const BACKEND = process.env.BACKEND_URL || "https://use-ai-malayalamai-production-ee70.up.railway.app";

export async function POST(req) {
  try {
    const form = await req.formData();
    const out  = new FormData();
    for (const [k, v] of form.entries()) out.append(k, v);
    const auth = req.headers.get("authorization");
    const headers = {};
    if (auth) headers["Authorization"] = auth;
    const upstream = await fetch(`${BACKEND}/audio/process`, {
      method: "POST", headers, body: out, cache: "no-store",
    });
    const ct   = upstream.headers.get("content-type") || "application/json";
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: { "content-type": ct, "cache-control": "no-store" },
    });
  } catch (err) {
    return NextResponse.json({ error: err.message || "Proxy failed" }, { status: 500 });
  }
}
